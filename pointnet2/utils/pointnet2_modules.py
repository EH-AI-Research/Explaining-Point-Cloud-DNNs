from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
import numpy as np
from pointnet2.utils import pointnet2_utils
import time

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.prev_xyz = None  # I've also put this into the forward call so that is is reinitialized in every call
        self.prev_indices = []
        self.xyz_delete_for_sampling_inds = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        #del self.prev_xyz
        #self.prev_xyz = None  # initialize in each forward pass

        # if self.xyz_sample_pool is None:
        #     self.xyz_sample_pool = xyz

        # alternative attempt to do progressive point dropping
        #xyz_flipped = xyz.transpose(1, 2).contiguous()
        #
        #if hasattr(self, 'drop_points_from_sampling'):  # and i == len(self.groupers)-1:
        #    # this function only has been added to the last set abstraction module
        #    # in the vote feature map before max and classification
        #    xyz_flipped = self.drop_points_from_sampling(xyz_flipped, self.npoint)

        if self.npoint is not None:
            if hasattr(self, 'save_features_progress'):
                # drop endpoints already catered for in the previous iteration by changing seed_xyz
                iter = self.iter
                if hasattr(self, 'prev_idx'):
                    prev_idx = self.prev_idx.type('torch.LongTensor')
                    if self.prev_xyz is None:
                        temp_xyz = xyz
                    else:
                        temp_xyz = self.prev_xyz
                    temp_xyz.data[0, prev_idx, :] = torch.zeros(1, 3,
                                                                requires_grad=False).cuda()  # changing all previous id to [0,0,0] xyz
                    # xyz_flipped = temp_xyz.transpose(1, 2).contiguous() #TODO discuss with team
                    xyz_flipped = xyz.transpose(1, 2).contiguous()
                    sample_inds = pointnet2_utils.furthest_point_sample(temp_xyz, self.npoint)
                    new_xyz = (pointnet2_utils.gather_operation(
                        xyz_flipped, sample_inds).transpose(1, 2).contiguous())
                    self.prev_xyz = temp_xyz
                else:
                    xyz_flipped = xyz.transpose(1, 2).contiguous()
                    sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                    new_xyz = (pointnet2_utils.gather_operation(
                        xyz_flipped, sample_inds).transpose(1, 2).contiguous())
                    self.prev_xyz = xyz
                    # sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            if hasattr(self, 'save_features_progress_input'):
                xyz_flipped = xyz.transpose(1, 2).contiguous()
                # xyz_sample_pool = xyz

                if self.ignore_for_sampling is not None:
                    delete_inds = self.ignore_for_sampling.data.cpu().numpy()
                    keep_inds = np.delete(np.arange(512), delete_inds, 0)
                    xyz_sample_pool = xyz.clone().data[:, keep_inds, :].cuda()

                    # sample_inds = pointnet2_utils.furthest_point_sample(temp_xyz, self.npoint)
                    # sample_inds = pointnet2_utils.furthest_point_sample(xyz_sample_pool, self.npoint)
                    # sample_inds = torch.tensor([np.sort(keep_inds, 0)[sample_inds.data.cpu().numpy()[0]]]).int().cuda()
                    if delete_inds.shape[0] < 384:
                        sample_inds = pointnet2_utils.furthest_point_sample(xyz_sample_pool, self.npoint)
                        sample_inds = torch.tensor([np.sort(keep_inds, 0)[sample_inds.data.cpu().numpy()[0]]]).int().cuda()
                    else:
                        sample_inds = torch.tensor([keep_inds[:128]]).int().cuda()
                else:
                    sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)

                new_xyz = (pointnet2_utils.gather_operation(
                    xyz_flipped, sample_inds).transpose(1, 2).contiguous())
                self.prev_xyz = new_xyz
                self.prev_xyz_indices = sample_inds
                # sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            else:
                if hasattr(self, 'prev_idx_input_progress'):
                    # prev_idx = self.prev_idx.type('torch.LongTensor')

                    if hasattr(self, 'old_drop_indecing_loop'):
                        if self.prev_xyz is None:
                            temp_xyz = xyz.clone()
                            # temp_xyz_np = temp_xyz.data.cpu().numpy()
                        else:
                            temp_xyz = self.prev_xyz

                        grad_xyz_np = self.grad_xyz.data.cpu().numpy().squeeze()
                        comp_xyz_np = xyz.data.cpu().numpy().squeeze()

                        indices = []
                        for i in range(0, len(grad_xyz_np)):
                            for j in range(0, len(comp_xyz_np)):
                                found_point = np.where(comp_xyz_np[j, :] == grad_xyz_np[i, :])
                                if len(found_point[0]) == 3 and j not in self.prev_indices:  # found_point[0].sum() == 3:
                                    indices.append(j)
                                    self.prev_indices.append(j)
                                    break
                        prev_idx = torch.Tensor(np.expand_dims(np.asarray(indices), 0)).type('torch.LongTensor').cuda()

                        # alternative 1: center explained points for sampling
                        temp_xyz.data[0, indices, :] = torch.zeros(1, 3, requires_grad=False).cuda()  # changing all previous id to [0,0,0] xyz
                        # new method
                        # temp_xyz.data[0, self.center_point_inds.long(), :] = torch.zeros(1, 3, requires_grad=False).cuda()  # changing all previous id to [0,0,0] xyz

                        sample_inds = pointnet2_utils.furthest_point_sample(temp_xyz, self.npoint)
                        self.prev_xyz = temp_xyz # [0, self.center_point_inds.long(), :]
                    else:
                        # alternative 2: remove explained points for sampling
                        temp_xyz = xyz.clone()

                        if self.xyz_delete_for_sampling_inds is None or self.xyz_delete_for_sampling_inds.size()[0] < 1536:
                            # delete the previous explained
                            if self.xyz_delete_for_sampling_inds is None:
                                self.xyz_delete_for_sampling_inds = self.center_point_inds
                            else:  # concat
                                self.xyz_delete_for_sampling_inds = torch.cat((self.xyz_delete_for_sampling_inds, self.center_point_inds))

                        delete_inds = self.xyz_delete_for_sampling_inds.cpu().data.numpy()
                        keep_inds = np.delete(np.arange(2048), delete_inds, 0)
                        xyz_sample_pool = temp_xyz.data[:, keep_inds, :].cuda()

                        sample_inds = pointnet2_utils.furthest_point_sample(xyz_sample_pool, self.npoint)
                        sample_inds = torch.tensor([np.sort(keep_inds, 0)[sample_inds.data.cpu().numpy()[0]]]).int().cuda()
                else:
                    sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)

                self.prev_xyz_indices = sample_inds

                # sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                xyz_flipped = xyz.transpose(1,2).contiguous()
                new_xyz = (pointnet2_utils.gather_operation(
                    xyz_flipped, sample_inds).transpose(1, 2).contiguous())
        else:
            sample_inds = None
            new_xyz = None

        """new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )"""

        #print("before loop")
        # if 'temp_xyz' in locals():
        #     test0 = temp_xyz.data.cpu().numpy()
        #     zero_counter = 0
        #     pasdas = test0[0]
        #     for x in test0[0]:
        #         if x[0] == 0. and x[1] == 0. and x[2] == 0.:
        #             zero_counter = zero_counter + 1
        #     print("zero", zero_counter)
        #
        # test1 = xyz.data.cpu().numpy()
        # if new_xyz is not None:
        #     test2 = new_xyz.data.cpu().numpy()

        new_features_list = []

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            if hasattr(self, 'save_features') :#and i == len(self.groupers)-1:
                # this function only has been added to the last set abstraction module
                # in the vote feature map before max and classification
                self.save_features(new_features, new_xyz, xyz)

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), sample_inds


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
