import import_base_path
import torch
from dgcnn.model import DGCNN
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps
from saliency_maps.saliency_method import GradCamWrapper as Saliency

if __name__ == '__main__':
    args = get_args('dgcnn')

    def get_classifier(num_points):
        classifier = DGCNN(args, output_channels=16)
        if args.use_cuda:
            classifier.cuda()
        classifier.load_state_dict(torch.load(args.model))
        classifier.eval()
        return classifier

    saliency = Saliency(get_classifier=get_classifier, cuda=args.use_cuda)
    visualize_heatmaps(saliency,
                       args.point_path,
                       num_points=args.num_points)