import import_base_path
import torch
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from grad_cam.grad_cam_pointnet2 import GradCamThirdModule
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('pn2')

    classifier = Pointnet2(input_channels=0, num_classes=2, use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam = GradCamThirdModule(classifier, k=3, target_layer_names=["2"], cuda=args.use_cuda)
    visualize_heatmaps(grad_cam,
                       args.point_path,
                       num_points=args.num_points)