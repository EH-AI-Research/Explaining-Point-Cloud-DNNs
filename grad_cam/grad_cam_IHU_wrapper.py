import numpy as np
import torch
import math

class IterativeHeatmapping:

    def __init__(
            self,
            grad_cam,
            step_size=25,
            value_bin_size=50):
        self.grad_cam = grad_cam
        self.step_size = step_size
        self.value_bin_size = value_bin_size

    def __call__(self, input, target_idx, high_drop=False, fixed_distribution=False):
        # copy input
        points = input.clone().detach()

        num_points = points.size()[2]  # 2048
        iterative_cam = np.zeros(num_points)
        cam = None

        # cam_add_gate is necessary to ignore the cam values of the already centered points
        cam_add_gate = np.zeros(num_points, dtype=np.dtype(np.float32))

        num_bins = math.ceil(num_points / self.value_bin_size)
        bin_distance = 1.0 / (num_bins - 1)  # - 1 because we have a bin at 0 and 1 as well
        print("num_bins", num_bins, "bin_distance", bin_distance)

        for i in range(num_points):
            # update heat map every self.step_size
            if i % self.step_size == 0:
                cam = self.grad_cam(points, target_idx)
                if np.isnan(cam).any():
                    cam = np.zeros(cam.size)
                cam = cam + cam_add_gate
                #result = self.grad_cam.classifier_output

            # get highest/lowe value from heatmap
            idx = np.argmax(cam) if high_drop else np.argmin(cam)

            # shift point to center
            points[:, :, idx] = 0

            # calculate grad value
            if fixed_distribution:
                bin_idx = math.floor(i / num_bins)
                iterative_cam[idx] = 1.0 - (bin_idx * bin_distance) if high_drop else 0.0 + (bin_idx * bin_distance)
            else:
                # take cam value
                confidence = torch.nn.functional.softmax(self.grad_cam.classifier_output).cpu()[0, target_idx]
                iterative_cam[idx] = cam[idx] #* ((num_points-i)/num_points) # * confidence

            # set cam value to minus (highdrop) / plus 1.5 (lowdrop) so that the next argmax/argmin call returns the next lower/higher value
            cam[idx] = -1.5 if high_drop else 1.5
            cam_add_gate[idx] = -2.5 if high_drop else 1.5

        if not fixed_distribution:
            iterative_cam = np.maximum(iterative_cam, 0)  # ReLU
            iterative_cam = iterative_cam - np.min(iterative_cam)
            if np.max(iterative_cam) > 0:
                # in the drop point experiment the gradients are at some time 0 leading to a division by zero error
                iterative_cam = iterative_cam / np.max(iterative_cam)

        return iterative_cam


class IterativeHeatmappingAccum:

    def __init__(
            self,
            grad_cam,
            step_size=25,
            min_point=500,
            drop_type='low',
            normalize_explicetly=True,
            acc_type='max',
            growing_weight=False,
            reverse_weight_growth=False,
            start_weight=0.9,
            disable_relu=False):  # 125 + 125 + 48
        assert drop_type in ['high', 'low', 'random']
        assert acc_type in ['average', 'max', 'min']
        self.grad_cam = grad_cam
        self.step_size = step_size
        self.min_point = min_point
        self.drop_type = drop_type
        self.normalize_explicetly = normalize_explicetly
        self.acc_type = acc_type
        self.growing_weight = growing_weight
        self.start_weight = start_weight
        self.reverse_weight_growth = reverse_weight_growth
        self.disable_relu = disable_relu

    def __call__(self, input, target_idx):
        points = input.clone().detach()

        num_points = points.size()[2]  # 2048
        cams = np.zeros(num_points)
        weights = np.zeros(num_points)

        num_bins = math.ceil((num_points - self.min_point) / self.step_size)

        # cam_gate is necessary to ignore the cam values of the already centered points
        cam_gate = np.ones(num_points, dtype=np.dtype(np.float32))
        cam_gate_changed = np.zeros(num_points, dtype=np.dtype(np.float32))

        if self.drop_type == 'random':
            rand_idx = np.random.choice(points.size(2), size=num_points, replace=False)

        for i in range(num_points):
            if i > num_points - self.min_point:
                break

            if i % self.step_size == 0:
                # update heat map
                cam = self.grad_cam(points, target_idx)
                if i == 0:
                    self.classifier_output = self.grad_cam.classifier_output
                if np.isnan(cam).any():
                    cam = np.zeros(cam.shape)

                if self.growing_weight:
                    if self.reverse_weight_growth:
                        # (1, self.start)
                        cam = cam * (1 - (i / self.step_size) * ((1 - self.start_weight) / num_bins))
                    else:
                        # (self.start, 1)
                        cam = cam * (self.start_weight + (i / self.step_size) * ((1 - self.start_weight) / num_bins))

                # update accumulated cam
                if self.acc_type == 'average':
                    cams = cams + (cam * cam_gate)
                    weights = weights + cam_gate
                elif self.acc_type == 'max':
                    cam_heat = cam - (cam_gate_changed * 100000)
                    cams = np.maximum(cams, cam_heat)
                elif self.acc_type == 'min':
                    cam_heat = cam + (cam_gate_changed * 100000)
                    cams = np.minimum(cams, cam_heat)

            # get highest/lowest/random value from heatmap
            idx = None
            if self.drop_type == 'high':
                cam_for_dropping = cam * cam_gate
                idx = np.argmax(cam_for_dropping)
            elif self.drop_type == 'low':
                cam_for_dropping = cam + (cam_gate_changed * 1000)
                idx = np.argmin(cam_for_dropping)
            else:  # random
                idx = rand_idx[:, i]

            # shift point to center
            points[:, :, idx] = 0
            cam_gate[idx] = 0
            cam_gate_changed[idx] = 1

        if self.acc_type == 'average':
            #print("average => cams / weights")
            cam = cams / weights
        else:
            #print("max or min")
            cam = cams

        if self.normalize_explicetly:
            if not self.disable_relu:
                cam = np.maximum(cam, 0)  # ReLU
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                # in the drop point experiment the gradients are at some time 0 leading to a division by zero error
                cam = cam / np.max(cam)

        return cam


class IterativeHeatmappingAccumParallel:

    def __init__(
            self,
            grad_cam,
            step_size=25,
            min_point=500,
            drop_type='low',
            normalize_explicetly=True,
            acc_type='max',
            growing_weight=False,
            reverse_weight_growth=False,
            start_weight=0.9,
            disable_relu=False):  # 125 + 125 + 48
        assert drop_type in ['high', 'low', 'random']
        assert acc_type in ['average', 'max', 'min']
        self.grad_cam = grad_cam
        self.step_size = step_size
        self.min_point = min_point
        self.drop_type = drop_type
        self.normalize_explicetly = normalize_explicetly
        self.acc_type = acc_type
        self.growing_weight = growing_weight
        self.start_weight = start_weight
        self.reverse_weight_growth = reverse_weight_growth
        self.disable_relu = disable_relu

    def __call__(self, input, target_idx):
        points = input.clone().detach()

        batch_size = points.size()[0]
        num_points = points.size()[2]  # 2048
        cams = np.zeros((batch_size, num_points))
        weights = np.zeros((batch_size, num_points))

        num_bins = math.ceil((num_points - self.min_point) / self.step_size)

        # cam_gate is necessary to ignore the cam values of the already centered points
        cam_gate = np.ones((batch_size, num_points), dtype=np.dtype(np.float32))
        cam_gate_changed = np.zeros((batch_size, num_points), dtype=np.dtype(np.float32))

        if self.drop_type == 'random':
            rand_idx = np.random.choice((batch_size, num_points), size=num_points, replace=False)

        for i in range(num_points):
            if i > num_points - self.min_point:
                break

            if i % self.step_size == 0:
                # update heat map
                cam = self.grad_cam(points, target_idx)
                if i == 0:
                    self.classifier_output = self.grad_cam.classifier_output
                if np.isnan(cam).any():
                    cam = np.zeros(cam.shape)

                if self.growing_weight:
                    if self.reverse_weight_growth:
                        # (1, self.start)
                        cam = cam * (1 - (i / self.step_size) * ((1 - self.start_weight) / num_bins))
                    else:
                        # (self.start, 1)
                        cam = cam * (self.start_weight + (i / self.step_size) * ((1 - self.start_weight) / num_bins))

                # update accumulated cam
                if self.acc_type == 'average':
                    cams = cams + (cam * cam_gate)
                    weights = weights + cam_gate
                elif self.acc_type == 'max':
                    cam_heat = cam - (cam_gate_changed * 1000)
                    cams = np.maximum(cams, cam_heat)
                elif self.acc_type == 'min':
                    cam_heat = cam + (cam_gate_changed * 1000)
                    cams = np.minimum(cams, cam_heat)

            # get highest/lowest/random value from heatmap
            idx = None
            if self.drop_type == 'high':
                cam_for_dropping = cam * cam_gate
                idx = np.argmax(cam_for_dropping, axis=1)
            elif self.drop_type == 'low':
                cam_for_dropping = cam + (cam_gate_changed * 1000)
                idx = np.argmin(cam_for_dropping, axis=1)
            else:  # random
                idx = rand_idx[i]

            # shift point to center
            for idx_indice in range(idx.size):
                points[idx_indice, :, idx[idx_indice]] = 0
                cam_gate[idx_indice, idx[idx_indice]] = 0
                cam_gate_changed[idx_indice, idx[idx_indice]] = 1

        if self.acc_type == 'average':
            #print("average => cams / weights")
            cam = cams / weights
        else:
            #print("max or min")
            cam = cams

        if self.normalize_explicetly:
            if not self.disable_relu:
                cam = np.maximum(cam, 0)  # ReLU
            cam = cam - np.expand_dims(np.min(cam, axis=1), 0).T
            if np.max(cam) > 0:
                # in the drop point experiment the gradients are at some time 0 leading to a division by zero error
                cam = cam / np.expand_dims(np.max(cam, axis=1), 0).T

        return cam