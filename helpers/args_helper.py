import argparse
import torch

default_models = {
    'pn1': './pre-trained/cls_model_249.pth',
    'pn2': './pre-trained/cls_model_pointnet2_249.pth', # cls_model_pointnet2_136.pth
    'dgcnn': './pre-trained/cls_model_249_dgcnn.pth',
}

def get_args(model):
    assert model in ['pn1', 'pn2', 'dgcnn']

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--disable_cuda', action='store_true', default=False, help='Do explicitely not use NVIDIA GPU acceleration')
    parser.add_argument('--point-path', type=str,
                        default='./shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                        help='Input point cloud path')
    parser.add_argument('--set_size', type=int, default=64, help='size of batch of point cloud to color')
    parser.add_argument('--model', type=str,
                        default=default_models[model],
                        help='model path')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of Points')

    # DGCNN Parameters
    if model == 'dgcnn':
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout rate')
        parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                            help='Dimension of embeddings')
        parser.add_argument('--k', type=int, default=20, metavar='N',
                            help='Num of nearest neighbors to use')

    # dropping experiment
    parser.add_argument('--n_samples', type=int, default=2874, help='Number of Point Cloud samples to use for the point dropping experiment')
    parser.add_argument('--n_drops', type=int, default=2048, help='Number of Points to drop during the dropping experiment')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the test samples for the point dropping experiment to make sure that all classes are present in cas n_samples is less than 2874')

    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    if args.n_samples > 2874 or args.n_samples <= 0:
        print("Required: 1 <= n_samples<= 2874")

    return args