import import_base_path
import numpy as np
import random
from pointnet.dataset import ShapeNetDataset
from grad_cam.grad_cam_IHU_wrapper import IterativeHeatmappingAccum
import time
import torch
from helpers.point_dropping_experiment_single import PointDropExperiment
from dgcnn.model import DGCNN
from grad_cam.grad_cam_dgcnn import GradCamDGCNN
from helpers.args_helper import get_args

if __name__ == '__main__':
    args = get_args('dgcnn')

    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    test_dataset = ShapeNetDataset(
        root=args.point_path,
        split='test',
        classification=True,
        npoints=args.num_points,
        data_augmentation=False,
        unique=False,
        #unique_path="./unique-network-select")  # unique-network-comparison  unique-tables
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    show_visualization = False

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    print("len(test_dataset.classes)", len(test_dataset.classes))
    classifier = DGCNN(args, output_channels=len(test_dataset.classes))
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()

    grad_cam = GradCamDGCNN(args, model=classifier, counterfactual=False, normalize=True, disable_relu=False, use_cuda=args.use_cuda)

    iterative_heatmapping = IterativeHeatmappingAccum(
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

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=classifier,
        grad_cam=iterative_heatmapping,
        testdataloader=testdataloader,
        num_drops=args.n_drops,
        steps=50,
        num_iterations=args.n_samples,  # max is 2874 = test set size
        update_cam=False,
        show_visualization=show_visualization,
        file_prefix='exp8-dgcnn-IHU-',
        plot_title_prefix_text="EXP8 Point GradCAM IHU DGCNN",
        create_png=True,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        print_progress=True,
        use_cuda=args.use_cuda,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)