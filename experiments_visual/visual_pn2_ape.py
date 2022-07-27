import import_base_path
import torch
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from grad_cam.grad_cam_pointnet2_APE import GradCamPointnet2
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('pn2')

    classifier = Pointnet2(input_channels=0, num_classes=16, use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam = GradCamPointnet2(args,
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
    visualize_heatmaps(grad_cam,
                       args.point_path,
                       num_points=args.num_points)