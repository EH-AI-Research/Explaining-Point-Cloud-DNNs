import types

import numpy as np
import torch
from torch.autograd import Variable
import open3d as o3d


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def __call__(self, x):
        outputs = []
        self.gradients = []
        x = x.transpose(1, 2)
        xyz, feats = self._break_up_pc(x)
        for name, module in self.model._modules.items():

            self.saved_features = None
            saved_xyz = xyz # before applying the module

            def save_features(sa_module_self, output, new_xyz, old_xyz):
                # features = output
                # features_center_xyz = new_xyz
                # features_old_xyz = old_xyz.transpose(1,2)
                # output.register_hook(save_gradients)
                #global saved_features
                self.saved_features = output
                output.register_hook(self.save_gradient)

            if name in self.target_layers:
                module.save_features = types.MethodType(save_features, module)
            xyz, feats, _sample_ids = module(xyz, feats)

            if name in self.target_layers:
                #feats.register_hook(self.save_gradient)
                xyz_target = saved_xyz
                outputs += [self.saved_features.squeeze(-1)]
            #if name == '7':
              #  x = torch.max(x, 2, keepdim=True)[0]
              #  x = x.view(-1, 1024)
        feats = feats.squeeze(-1)
        return outputs, feats, xyz_target


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output, xyz_target = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        # F.log_softmax(x, dim=1)
        return target_activations, output,xyz_target


class GradCamThirdModule:
    def __init__(self, model, target_layer_names,
                 interpolate=True, k=3,
                 counterfactual=False, disable_relu=False, cuda=True):
        self.model = model
        self.cuda = cuda
        self.extractor = ModelOutputs(self.model, target_layer_names)
        self.counterfactual = counterfactual
        self.disable_relu = disable_relu
        self.interpolate = interpolate
        self.k = k

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output, xyz_target = self.extractor(input.cuda())
        else:
            features, output, xyz_target = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        self.classifier_output = output

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # one_hot.backward(retain_variables=True)
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        grads_val = self.extractor.get_gradients()[-1].view(1, 1024, -1).cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1

        target = features[-1].view(1, 1024, -1)
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        if not self.disable_relu:
            cam = np.maximum(cam, 0)  # ReLU
        #cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        xyz_target = torch.transpose(xyz_target, 1, 2)#xyz_target.cpu().data.numpy()[0, :]
        self.xyz_target = xyz_target

        if self.interpolate:
            return self.upscale_mask(input, xyz_target, cam)

        return cam

    def upscale_mask(self, input, xyz_target, mask):
        mask_xyz = xyz_target.transpose(1, 0).view(3, -1).contiguous().data.cpu().data.numpy()
        points = input[0].transpose(1, 0).contiguous().data.cpu().numpy()

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(prc_r_all)
        pcd_tree = o3d.geometry.KDTreeFlann(mask_xyz)
        mask_upscale = np.zeros(len(points))

        mask_xyz = mask_xyz.T

        for i in range(0, len(points)):
            pt = points[i]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, self.k)

            if self.k == 1 or np.linalg.norm(mask_xyz[idx[0]] - pt) == 0:  # is the same, explanation already exists
                mask_upscale[i] = mask[idx[0]]
            else:
                weighted_cam_sum = 0
                weight_sum = 0
                for id in idx:
                    d = np.linalg.norm(mask_xyz[id] - pt)
                    weighted_cam_sum += (1.0 / d) * mask[id]
                    weight_sum += (1.0 / d)

                mask_upscale[i] = weighted_cam_sum / weight_sum

        return mask_upscale # return only the mask