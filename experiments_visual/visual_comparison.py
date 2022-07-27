import import_base_path
import torch
import open3d as o3d
from grad_cam.grad_cam_pointnet1 import GradCam
from grad_cam.grad_cam_IHU_wrapper import IterativeHeatmappingAccum
from pointnet.model import PointNetCls
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps
from dgcnn.model import DGCNN
from grad_cam.grad_cam_dgcnn import GradCamDGCNN
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from grad_cam.grad_cam_pointnet2 import GradCamThirdModule
from grad_cam.grad_cam_pointnet2_APE import GradCamPointnet2

if __name__ == '__main__':
    # Pointnet
    args = get_args('pn1')

    classifier = PointNetCls(k=16, num_points=args.num_points)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam_pn1 = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda)

    # Pointnet IHU
    grad_cam_pn1_ihu = IterativeHeatmappingAccum(
        grad_cam=grad_cam_pn1,
        drop_type='low',
        normalize_explicetly=False,
        disable_relu=False,
        acc_type='max',
        growing_weight=True,
        reverse_weight_growth=False,
        start_weight=0.75,
        step_size=10,
        min_point=20,
    )

    # DGCNN
    args = get_args('dgcnn')
    classifier = DGCNN(args, output_channels=16)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam_dgcnn = GradCamDGCNN(args, model=classifier, use_cuda=args.use_cuda)

    # DGCNN IHU
    args = get_args('dgcnn')

    grad_cam_dgcnn_ihu = IterativeHeatmappingAccum(
        grad_cam=grad_cam_dgcnn,
        drop_type='low',
        normalize_explicetly=False,
        disable_relu=False,
        acc_type='max',
        growing_weight=True,
        start_weight=0.8,
        step_size=25,
        min_point=100,
    )

    # Pointnet++ Interpolation
    args = get_args('pn2')

    classifier = Pointnet2(input_channels=0, num_classes=16, use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam_pn2_interpolate = GradCamThirdModule(classifier, k=3, target_layer_names=["2"], cuda=args.use_cuda)

    # Pointnet++ APE
    grad_cam_pn2_ape = GradCamPointnet2(args,
                                model=classifier,
                                module_n='1',
                                interp_mode=False,
                                progress_mode_input=True,
                                num_point_drops=2048,
                                progress_mode_input_plus_interpolate=True,
                                progress_mode_input_plus_interpolate_k=1,
                                progress_mode_input_old_droping_indecing_loop=True,
                                post_accumulation_normalization=False,
                                disable_relu=False)

    visualize_heatmaps([grad_cam_pn1, grad_cam_pn1_ihu, grad_cam_dgcnn,
                        grad_cam_dgcnn_ihu, grad_cam_pn2_interpolate, grad_cam_pn2_ape],
                       args.point_path,
                       num_points=args.num_points)