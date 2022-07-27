import open3d
import torch
from grad_cam.vanilla_gradient import Gradient
from pointnet.model import PointNetCls
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('pn1')

    classifier = PointNetCls(k=16, num_points=args.num_points)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    visualize_heatmaps(Gradient(classifier, 'absolute'),
                       args.point_path,
                       num_points=args.num_points)