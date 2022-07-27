import torch
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from grad_cam.vanilla_gradient import Gradient
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('pn2')

    classifier = Pointnet2(input_channels=0, num_classes=16, use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    visualize_heatmaps(Gradient(classifier, 'absolute'),
                       args.point_path,
                       num_points=args.num_points)