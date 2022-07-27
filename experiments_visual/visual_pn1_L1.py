import import_base_path
import torch
from grad_cam.grad_cam_pointnet1 import GradCam
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

    grad_cam = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda)
    visualize_heatmaps(grad_cam,
                       args.point_path,
                       num_points=args.num_points)