import types

import numpy as np
import open3d as o3d
import torch
from torch.autograd import Variable


class GradCamPointnet2:
    def __init__(self, args, model, module_n='1', interp_mode = False, progress_mode = False, progress_mode_input = False,
                 num_point_drops=2048, progress_mode_input_plus_interpolate=False, progress_mode_input_plus_interpolate_k=6,
                 progress_mode_input_old_droping_indecing_loop=False, counterfactual=False, post_accumulation_normalization=False,
                 disable_relu=False):
        self.args = args
        self.model = model
        self.points = None

        # self.gradients = []
        # self.features = []
        # self.features_xyz = []
        # self.features_center_xyz = []
        #
        # self.features_old_xyz = []
        # self.feature_idx = []  # will contain the indices of the cluster votes
        self.module_n = module_n
        if self.module_n == '0':
            self.num_points = 2048
        elif self.module_n == '1':
            self.num_points = 512
        else:
            assert False, "Oh no! The Module number is wrong!"

        self.num_point_drops = num_point_drops

        # self.accumulated_cam = np.zeros(self.num_points)
        # self.num_cam = np.zeros(self.num_points)
        #
        # if interp_mode:
        #     self.calculated_cam = np.zeros(128)

        self.interp_mode = interp_mode
        self.progress_mode = progress_mode
        self.progress_mode_input = progress_mode_input

        self.progress_mode_input_plus_interpolate = progress_mode_input_plus_interpolate
        self.progress_mode_input_plus_interpolate_k = progress_mode_input_plus_interpolate_k

        self.progress_mode_input_old_droping_indecing_loop = progress_mode_input_old_droping_indecing_loop

        self.counterfactual = counterfactual

        self.post_accumulation_normalization = post_accumulation_normalization

        self.disable_relu = disable_relu

        # if progress_mode or progress_mode_input:
        #     self.cam_progress = []
        #     self.grad_xyz_progress = []
        #     self.seed_xyz_new = []
        #     self.input_xyz_new = []

    def save_idx(self, grouper_self, idx):
        # feature_idx = idx
        self.feature_idx.append(idx)

    def save_features(self, ctx, output, new_xyz, old_xyz):
        self.features.append(output)
        self.features_center_xyz.append(new_xyz)
        self.features_old_xyz.append(old_xyz.transpose(1, 2))
        output.register_hook(self.save_gradients)

    def save_features_interp(self, ctx, output, xyz, idx):
        self.features = output
        self.features_xyz = xyz
        # self.features_center_xyz = new_xyz
        output.register_hook(self.save_gradients)
        self.feature_idx.append(idx)  # also saving features additionally

    def save_features_progress(self,ctx,output, xyz,idx):
        self.features = output
        self.features_xyz.append(xyz)
        #self.features_center_xyz = new_xyz
        output.register_hook(self.save_gradients)
        self.feature_idx.append(idx) # also saving features additionally

    def save_features_progress_input(self, ctx, output, xyz, idx):
        ##self.features = output
        self.features_xyz = xyz
        # self.features_center_xyz = new_xyz
        ##output.register_hook(self.save_gradients)
        self.feature_idx.append(idx)  # also saving features additionally

    def save_features_progress_input_mp(self, ctx, output, xyz, idx):
        self.features = output
        self.features_xyz = xyz
        # self.features_center_xyz = new_xyz
        output.register_hook(self.save_gradients)
        #self.feature_idx.append(idx)  # also saving features additionally

    def save_gradients(self, grad):
        self.gradients.append(grad)

    def accumulate_cam(self, inx):
        grads_val = self.gradients[inx].cpu().data.numpy()
        target = self.features[inx].cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :]
        if not self.disable_relu:
            cam = np.maximum(cam, 0)  # ReLU
        cam_flat = cam.flatten()
        idx_flat = self.feature_idx[inx].cpu().data.numpy().flatten()
        for i in range(idx_flat.size):
            id = idx_flat[i]
            self.accumulated_cam[id] += cam_flat[i]
            self.num_cam[id] += 1

    def calc_cam(self):
        grads_val = self.gradients[0].cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1

        target = self.features
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]
        if not self.disable_relu:
            cam = np.maximum(cam, 0)  # ReLU
        self.calculated_cam = cam

    def calc_cam_progress(self,inx):
        grads_val = self.gradients[inx].cpu().data.numpy()

        # necessary for MP approach
        grads_val = np.reshape(grads_val, (1, 1024, 128))

        if self.counterfactual:
            grads_val = grads_val * -1

        target = self.features
        target = target.cpu().data.numpy()[0, :]

        # necessary for MP appraoach
        target = np.reshape(target, (1024, 128))

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]
        if not self.disable_relu:
            cam = np.maximum(cam, 0)  # ReLU
        self.calculated_cam = cam
        return np.maximum(cam, 0)

    def __call__(self, input, target_idx=False):

        net = self.model

        # reset values
        self.gradients = []
        self.features = []
        self.features_xyz = []
        self.features_center_xyz = []

        self.features_old_xyz = []
        self.feature_idx = []  # will contain the indices of the cluster votes

        self.accumulated_cam = np.zeros(self.num_points)
        self.num_cam = np.zeros(self.num_points)

        if self.interp_mode:
            self.calculated_cam = np.zeros(128)

        if self.progress_mode or self.progress_mode_input:
            self.cam_progress = []
            self.grad_xyz_progress = []
            self.seed_xyz_new = []
            self.input_xyz_new = []


        #net.cuda()
        #net.load_state_dict(torch.load(self.args.model))
        #net.eval()  # eval not needed at this stage unless grad cam is commented

        if self.interp_mode is True:
            return self.run_interp_mode(input, net, target_idx)
        elif self.progress_mode is True:
            return self.run_progress_mode(input, net, target_idx)
        elif self.progress_mode_input is True:
            return self.run_progress_mode_input(input, net, target_idx)
        else:
            return self.run_accumulation_mode(input, net, target_idx)


    def run_interp_mode(self, input, net, target_idx):
        vis_model = net.features._modules[self.module_n]  # select the correct number
        vis_model.save_features_interp = types.MethodType(self.save_features_interp, vis_model)

        output = net(input)

        if target_idx:
            index = target_idx
        else:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)
        net.zero_grad()
        one_hot.backward(retain_graph=True)

        # Calculate Gradients
        self.calc_cam()

        cam_f = self.calculated_cam
        #cam_f[np.isnan(cam_f)] = 0
        cam_f = cam_f - np.min(cam_f)
        cam_f = cam_f / np.max(cam_f)

        return cam_f


    def run_progress_mode(self, input, net, target_idx):
        """Changing orignal network to enable gradient loggings"""

        vis_model = net.features._modules[self.module_n]  # select the correct number
        vis_model.save_features_progress = types.MethodType(self.save_features_progress, vis_model)
        for i in range(0, int(512/128)):  # 1024  1024/256 --> number of points in seeds /256 number of points in grad cam put
            vis_model.iter = i
            output = net(input)

            index = np.argmax(output.cpu().data.numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot.cuda() * output)
            net.zero_grad()
            one_hot.backward(retain_graph=True)

            # Calculate Gradients
            self.calc_cam_progress(inx = i)

            cam_f = self.calculated_cam
            cam_f[np.isnan(cam_f)] = 0
            cam_f = cam_f - np.min(cam_f)
            cam_f = cam_f / np.max(cam_f)
            cam_f[np.isnan(cam_f)] = 0

            self.cam_progress.append(cam_f)
            if i>0:
                xyz = vis_model.prev_xyz.detach()
                temp_xyz_np = xyz.cpu().data.numpy()
                self.seed_xyz_new.append(temp_xyz_np)
                #self.seed_xyz_new.append(xyz)
                #va.prev_xyz.detach()
                #temp_xyz_np = va.prev_xyz.cpu().data.numpy()
            else:
                xyz = vis_model.prev_xyz.detach()
                temp_xyz_np = xyz.cpu().data.numpy()
                self.seed_xyz_new.append(temp_xyz_np)
            vis_model.prev_idx = self.feature_idx[i]

        return cam_f


    def run_accumulation_mode(self, input, net, target_idx):
        vis_model = net.features._modules[self.module_n]  # select the correct number
        vis_model.save_features = types.MethodType(self.save_features, vis_model)

        for i in range(0, 3):
            grouper = net.features._modules[self.module_n].groupers._modules[str(i)]
            grouper.save_idx = types.MethodType(self.save_idx, grouper)

        output = net(input)
        self.classifier_output = output

        index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)
        net.zero_grad()
        one_hot.backward(retain_graph=True)

        self.gradients.reverse()  # reversing since the gradients are flipped

        # Starting to accumulate gradients
        for inx in range(0, 2):
            self.accumulate_cam(inx)

        cam_f = self.accumulated_cam / self.num_cam
        cam_f[np.isnan(cam_f)] = 0
        cam_f = cam_f - np.min(cam_f)
        cam_f = cam_f / np.max(cam_f)
        cam_f[np.isnan(cam_f)] = 0

        return cam_f


    ###########
    ########### This one!
    ###########
    def run_progress_mode_input(self, input, net, target_idx):
        vis_model = net.features._modules[self.module_n]  # select the correct number
        last_module = net.features._modules['2']
        vis_model.save_features_progress_input = types.MethodType(self.save_features_progress_input, vis_model)
        last_module.save_features = types.MethodType(self.save_features_progress_input_mp, last_module)

        # reset values for module 0 in each call
        net.features._modules['0'].prev_xyz = None
        net.features._modules['0'].prev_indices = []
        net.features._modules['0'].xyz_delete_for_sampling_inds = None
        #net.features._modules['0'].grad_xyz = None

        if self.progress_mode_input_old_droping_indecing_loop is True:
            net.features._modules['0'].old_drop_indecing_loop = True

        # reset, necessary fix if this object should be re-used for experiments isntead of initializing it ever ytime
        vis_model.prev_xyz = None
        net.features._modules['1'].ignore_for_sampling = None
        # if self.progress_mode_input_old_droping_indecing_loop is True:
        #     vis_model.old_sampling_approach = True

        index = target_idx

        for i in range(0, int(
                self.num_point_drops / 128)):  # 1024  1024/256 --> number of points in seeds /256 number of points in grad cam put
            vis_model.iter = i
            # if i == 15:
            #     print("stop")

            output = net(input)

            if not target_idx:
                index = np.argmax(output.cpu().data.numpy())

            if i == 0:
                # just save the output of the first run in self.classifier_output that it
                # can be used by the experiment class without having to do another forward call
                self.classifier_output = output

            # one_hot_np_debug = output.cpu().data.numpy()

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot.cuda() * output)
            net.zero_grad()
            one_hot.backward(retain_graph=True)

            # highest_index = np.argmax(one_hot_np_debug)
            # total_output_activation_value = one_hot_np_debug.sum()
            # activation_value = one_hot_np_debug[0][index]
            # #confidence = activation_value / total_output_activation_value
            # confidence = torch.nn.functional.softmax(output).cpu()[0, index]
            # print("ouput", one_hot_np_debug)
            # print("total output activations", one_hot_np_debug.sum())
            # print("target output activation", activation_value)
            # print("confidence", confidence)

            # Calculate Gradients
            cam_f = self.calc_cam_progress(inx=i)
            # all_cam = cam_f.sum()
            cam_f[np.isnan(cam_f)] = 0

            if self.post_accumulation_normalization is False:
                cam_f = cam_f - np.min(cam_f)
                cam_f = cam_f / np.max(cam_f)
                cam_f[np.isnan(cam_f)] = 0

            grad_xyz = vis_model.prev_xyz#.detach()

            xyz_unique = np.unique(grad_xyz.data.cpu().numpy(), axis=1)
            if xyz_unique.shape[1] < 128/3*2:
                # stop immediately
                break

            self.grad_xyz_progress.append(grad_xyz)
            self.cam_progress.append(cam_f)
            #grad_xyz_np = grad_xyz.detach().cpu().data.numpy()

            #grad_xyz = pointnet2_utils.gather_operation(xyz_flipped, self.feature_idx[i]).transpose(1, 2).contiguous()


            #vis_model.prev_idx = self.feature_idx[i]

            net.features._modules['0'].prev_idx_input_progress = True
            net.features._modules['0'].grad_xyz = grad_xyz  # required for old approach

            # get indices of points which need be set to zero
            if self.progress_mode_input_old_droping_indecing_loop is False:
                first_sample_inds = net.features._modules['0'].prev_xyz_indices[0]
                second_sample_inds = net.features._modules['1'].prev_xyz_indices[0]
                exaplained_inds = torch.index_select(first_sample_inds, 0, second_sample_inds.long())
                net.features._modules['0'].center_point_inds = exaplained_inds

                if i >= 12:
                    if net.features._modules['1'].ignore_for_sampling is None:
                        net.features._modules['1'].ignore_for_sampling = second_sample_inds
                    else:
                        net.features._modules['1'].ignore_for_sampling = torch.cat((net.features._modules['1'].ignore_for_sampling, second_sample_inds), axis=0)

                # if i>0:
            #     temp_xyz = net.features._modules['0'].prev_xyz # Seed XYZ New
            #     temp_xyz_np = temp_xyz.cpu().data.numpy()
            #
            #     self.input_xyz_new.append(temp_xyz_np)


        # agregate
        cam_f_list = self.cam_progress
        xyz_list = self.grad_xyz_progress

        # Aggregate CAM
        xyz_aggregate = xyz_list[0]
        cam_aggregate = cam_f_list[0]

        for i in range(1, len(xyz_list)):
            xyz_aggregate = torch.cat((xyz_aggregate, xyz_list[i]), 1)
            cam_aggregate = np.concatenate((cam_aggregate, cam_f_list[i]), 0)

        self.cam_aggregate = cam_aggregate
        self.xyz_aggregate = xyz_aggregate

        if self.progress_mode_input_plus_interpolate:
            pcd = o3d.geometry.PointCloud()
            xyz_aggregate_np = self.xyz_aggregate.data.cpu().numpy()[0]
            pcd.points = o3d.utility.Vector3dVector(xyz_aggregate_np)

            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            input_np_flipped = input.data.cpu().numpy()[0].T
            cam_input = np.zeros(input_np_flipped.shape[0])

            for i in range(input_np_flipped.shape[0]):
                pt = input_np_flipped[i]
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, self.progress_mode_input_plus_interpolate_k)
                # [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 7)

#
                if self.progress_mode_input_plus_interpolate_k == 1 or np.linalg.norm(xyz_aggregate_np[idx[0]] - pt) == 0:
                    cam_input[i] = cam_aggregate[idx[0]]
                    continue

                # apply rervse distance weighted interpolation
                # http://www.gitta.info/ContiSpatVar/de/html/Interpolatio_learningObject2.html
                weighted_cam_sum = 0
                weight_sum = 0
                min_distance = 0.01 # 0.05  # previous was 0.1
                # if self.progress_mode_input_plus_interpolate_k > 1:
                #     min_distance = (np.linalg.norm(xyz_aggregate_np[idx[1]] - pt) / 2.0).astype('float32')
                for id in idx:
                    # d = np.around(max(min_distance, np.linalg.norm(xyz_aggregate_np[id] - pt)), decimals=6)  # min distance 0.1
                    d = max(min_distance, np.linalg.norm(xyz_aggregate_np[id] - pt))  # min distance 0.1
                    # d = np.linalg.norm(xyz_aggregate_np[id] - pt)
                    weighted_cam_sum += (1.0 / d) * cam_aggregate[id]
                    weight_sum += (1.0 / d)

                cam_input[i] = weighted_cam_sum / weight_sum

            self.cam_aggregate = cam_input
            self.xyz_aggregate = input.transpose(1, 2)

        if self.post_accumulation_normalization is True:
            cam_f = self.cam_aggregate
            cam_f = cam_f - np.min(cam_f)
            cam_f = cam_f / np.max(cam_f)
            cam_f[np.isnan(cam_f)] = 0
            self.cam_aggregate = cam_f

        self.points = self.xyz_aggregate.transpose(1, 2)
        return self.cam_aggregate
