import import_base_path
import torch
from dgcnn.model import DGCNN
from grad_cam.grad_cam_dgcnn import GradCamDGCNN
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('dgcnn')

    classifier = DGCNN(args, output_channels=16)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam = GradCamDGCNN(args, model=classifier, use_cuda=args.use_cuda)
    visualize_heatmaps(grad_cam,
                       args.point_path,
                       num_points=args.num_points)