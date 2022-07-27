import open3d
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class SphereAttack():
    def __init__(self, num_drop, num_steps, num_classes, get_classifier, drop_pos=True, power=1, cuda=False):
        self.a = num_drop  # how many points to remove
        self.k = num_steps

        self.count = np.zeros((num_classes,), dtype=bool)
        self.all_counters = np.zeros((num_classes, 3), dtype=int)

        #self.model = model
        self.get_classifier = get_classifier
        self.classifiers = {}

        self.drop_pos = drop_pos
        self.power = power  # x: -dL/dr*r^x
        self.grad = None
        self.cuda = cuda

    def save_gradient(self, input_grad):
        self.grad = input_grad

    def ihu_heatmap(self, input, target):
        input_pc = input.detach().clone()
        num_points = input_pc.size()[2]
        heatmap_values = np.arange(num_points) / num_points
        if self.drop_pos:
            heatmap_values = np.flip(heatmap_values)
        heatmap_value_pointer = 0
        heatmap = np.zeros(num_points)  # assume B=1 for now
        original_idx = np.arange(num_points)
        first_pred = None

        for i in range(self.k):
            input_pc = Variable(input_pc, requires_grad=True)
            input_pc.register_hook(self.save_gradient)

            #num_points = input_pc.size()[2]
            if num_points in self.classifiers:
                self.model = self.classifiers[num_points]
            else:
                self.model = self.get_classifier(input_pc.size()[2])
            self.model.zero_grad()
            pred = self.model(input_pc)
            if first_pred is None:
                first_pred = pred
            loss = F.nll_loss(pred, target)
            # if opt.feature_transform:
            #   loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()

            grad = self.grad.detach().transpose(1, 2).cpu().numpy()

            input_pc = input_pc.detach().transpose(1, 2).cpu().numpy()
            sphere_core = np.median(input_pc, axis=1, keepdims=True)

            sphere_r = np.sqrt(np.sum(np.square(input_pc - sphere_core), axis=2))  ## BxN

            sphere_axis = input_pc - sphere_core  ## BxNx3

            if self.drop_pos:
                sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                          np.power(sphere_r, self.power))
            else:
                sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                         np.power(sphere_r, self.power))

            drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1] - self.a, axis=1)[:, -self.a:]

            tmp = np.zeros((input_pc.shape[0], input_pc.shape[1] - self.a, 3), dtype=float)
            for j in range(input_pc.shape[0]):  # along B
                tmp[j] = np.delete(input_pc[j], drop_indice[j], axis=0)  # along N points to delete

            # assume B=1 for now
            for didx in np.flip(drop_indice[0]):
                heatmap[original_idx[didx]] = heatmap_values[heatmap_value_pointer]
                heatmap_value_pointer += 1

            original_idx = np.delete(original_idx, drop_indice[0], axis=0)  # along N points to delete

            input_pc_np = tmp.copy()
            if self.cuda:
                input_pc = torch.from_numpy(input_pc_np).cuda().type_as(input).transpose(1, 2)
            else:
                input_pc = torch.from_numpy(input_pc_np).type_as(input).transpose(1, 2)
            num_points -= self.a

        #np.argsort

        return heatmap, first_pred

    def heatmap(self, input, target):
        input_pc = input.detach().clone()
        num_points = input_pc.size()[2]
        heatmap_values = np.arange(num_points) / num_points
        if self.drop_pos:
            heatmap_values = np.flip(heatmap_values)
        heatmap_value_pointer = 0
        heatmap = np.zeros(num_points)  # assume B=1 for now
        original_idx = np.arange(num_points)

        input_pc = Variable(input_pc, requires_grad=True)
        input_pc.register_hook(self.save_gradient)

        #num_points = input_pc.size()[2]
        if num_points in self.classifiers:
            self.model = self.classifiers[num_points]
        else:
            self.model = self.get_classifier(input_pc.size()[2])
        self.model.zero_grad()
        pred = self.model(input_pc)
        loss = F.nll_loss(pred, target)
        # if opt.feature_transform:
        #   loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()

        grad = self.grad.detach().transpose(1, 2).cpu().numpy()

        input_pc = input_pc.detach().transpose(1, 2).cpu().numpy()
        sphere_core = np.median(input_pc, axis=1, keepdims=True)

        sphere_r = np.sqrt(np.sum(np.square(input_pc - sphere_core), axis=2))  ## BxN

        sphere_axis = input_pc - sphere_core  ## BxNx3

        if self.drop_pos:
            sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                      np.power(sphere_r, self.power))
        else:
            sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                     np.power(sphere_r, self.power))

        drop_indice = np.flip(np.argpartition(sphere_map, kth=sphere_map.shape[1] - self.a, axis=1)[0])  # [:, -self.a:]

        # assume B=1 for now
        for didx in np.flip(drop_indice):
            heatmap[original_idx[didx]] = heatmap_values[heatmap_value_pointer]
            heatmap_value_pointer += 1

        return heatmap, pred

    def drop_points(self, input, target):
        pointclouds_pl_adv = input.detach().clone()

        self.grad = None

        for i in range(self.k):
            # run model mit pointclouds_pl_adv and get gradients
            pointclouds_pl_adv = Variable(pointclouds_pl_adv, requires_grad=True)
            pointclouds_pl_adv.register_hook(self.save_gradient)

            num_points = pointclouds_pl_adv.size()[2]
            if num_points in self.classifiers:
                self.model = self.classifiers[num_points]
            else:
                self.model = get_classifier(pointclouds_pl_adv.size()[2])
            self.model.zero_grad()
            pred = self.model(pointclouds_pl_adv)
            loss = F.nll_loss(pred, target.unsqueeze(0))
            # if opt.feature_transform:
            #   loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()

            grad = self.grad.detach().transpose(1, 2).cpu().numpy()
            #print("grad", grad)

            # change the grad into spherical axis and compute r*dL/dr
            ## mean value
            # sphere_core = np.sum(pointclouds_pl_adv, axis=1, keepdims=True)/float(pointclouds_pl_adv.shape[1])
            ## median value
            pointclouds_pl_adv = pointclouds_pl_adv.detach().transpose(1, 2).cpu().numpy()
            sphere_core = np.median(pointclouds_pl_adv, axis=1, keepdims=True)

            sphere_r = np.sqrt(np.sum(np.square(pointclouds_pl_adv - sphere_core), axis=2))  ## BxN

            sphere_axis = pointclouds_pl_adv - sphere_core  ## BxNx3

            if FLAGS.drop_neg:
                sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                         np.power(sphere_r, FLAGS.power))
            else:
                sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
                                          np.power(sphere_r, FLAGS.power))

            drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1] - self.a, axis=1)[:, -self.a:]

            tmp = np.zeros((pointclouds_pl_adv.shape[0], pointclouds_pl_adv.shape[1] - self.a, 3), dtype=float)
            for j in range(pointclouds_pl.shape[0]):
                tmp[j] = np.delete(pointclouds_pl_adv[j], drop_indice[j], axis=0)  # along N points to delete

            pointclouds_pl_adv_np = tmp.copy()
            if self.cuda:
                pointclouds_pl_adv = torch.from_numpy(pointclouds_pl_adv_np).cuda().type_as(input).transpose(1, 2)
            else:
                pointclouds_pl_adv = torch.from_numpy(pointclouds_pl_adv_np).type_as(input).transpose(1, 2)

        return pointclouds_pl_adv_np


class GradCamWrapper:
    def __init__(self, get_classifier, num_classes=16, num_drop_points=1000, step_size=5, heatmap_updates=False, cuda=False):
        self.classifier_output = None
        self.get_classifier = get_classifier
        self.num_classes = num_classes
        self.num_drop_points = num_drop_points
        self.step_size = step_size
        self.heatmap_updates = heatmap_updates
        self.cuda = cuda

    def __call__(self, input, target, low_drop=False):
        num_steps = int(self.num_drop_points / self.step_size)
        attack = SphereAttack(self.step_size, num_steps, get_classifier=self.get_classifier, num_classes=self.num_classes,
                              drop_pos=(not low_drop), cuda=self.cuda)
        if self.heatmap_updates:
            heatmap, pred = attack.ihu_heatmap(input, target)
        else:
            heatmap, pred = attack.heatmap(input, target)
        self.classifier_output = pred
        #evaluate(num_votes=1, model=get_classifier(2048), input=input, label=target, num_drop=5, num_steps=20,
        #         num_point=2048, num_classes=2, get_classifier=get_classifier)
        #print("Next")
        return heatmap