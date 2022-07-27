import import_base_path
import torch
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from saliency_maps.saliency_method import GradCamWrapper as Saliency
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('pn2')

    def get_classifier(num_points):
        classifier = Pointnet2(input_channels=0, num_classes=16, use_xyz=True)
        if args.use_cuda:
            classifier.cuda()
        classifier.load_state_dict(torch.load(args.model))
        classifier.eval()
        return classifier

    saliency = Saliency(get_classifier=get_classifier, cuda=args.use_cuda)
    visualize_heatmaps(saliency,
                       args.point_path,
                       num_points=args.num_points)