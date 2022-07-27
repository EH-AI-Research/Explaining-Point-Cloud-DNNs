import import_base_path
import torch
from grad_cam.grad_cam_pointnet1 import GradCam
from pointnet.model import PointNetCls
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps
from saliency_maps.saliency_method import GradCamWrapper as Saliency

if __name__ == '__main__':
    args = get_args('pn1')

    def get_classifier(num_points):
        classifier = PointNetCls(k=16, num_points=num_points)
        if args.use_cuda:
            classifier.cuda()
        classifier.load_state_dict(torch.load(args.model))
        classifier.eval()
        return classifier

    saliency = Saliency(get_classifier=get_classifier, cuda=args.use_cuda)
    visualize_heatmaps(saliency,
                       args.point_path,
                       num_points=args.num_points)