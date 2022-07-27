import import_base_path
import torch
from dgcnn.model import DGCNN
from grad_cam.grad_cam_dgcnn import GradCamDGCNN
from grad_cam.grad_cam_IHU_wrapper import IterativeHeatmappingAccum
from helpers.args_helper import get_args
from helpers.visual_helper import visualize_heatmaps

if __name__ == '__main__':
    args = get_args('dgcnn')

    classifier = DGCNN(args, output_channels=16)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam = GradCamDGCNN(args, model=classifier, normalize=True, use_cuda=args.use_cuda)
    ihu_grad_cam = IterativeHeatmappingAccum(
        grad_cam=grad_cam,
        drop_type='low',
        normalize_explicetly=False,
        disable_relu=False,
        acc_type='max',
        growing_weight=True,
        start_weight=0.8,
        step_size=25,
        min_point=100,
    )

    visualize_heatmaps(ihu_grad_cam,
                       args.point_path,
                       num_points=args.num_points)