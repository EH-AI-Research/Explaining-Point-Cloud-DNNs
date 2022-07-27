import import_base_path
import numpy as np
import random
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
from grad_cam.grad_cam_pointnet1 import GradCam
from helpers.args_helper import get_args
from grad_cam.grad_cam_IHU_wrapper import IterativeHeatmappingAccum
import time
import torch
from helpers.point_dropping_experiment_single import PointDropExperiment

if __name__ == '__main__':
    args = get_args('pn1')

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
        #unique_path='./unique-qualitative-point-dropping-table'
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    show_visualization = False

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    print("len(test_dataset.classes)", len(test_dataset.classes))
    classifier = PointNetCls(k=len(test_dataset.classes), num_points=args.num_points)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))


    grad_cam = GradCam(model=classifier, target_layer_names=["7"], counterfactual=False, use_cuda=args.use_cuda,
                       normalize=False, disable_relu=False)

    iterative_heatmapping = IterativeHeatmappingAccum(
        grad_cam=grad_cam,
        drop_type='low',
        normalize_explicetly=True,
        disable_relu=False,
        acc_type='max',
        growing_weight=True,
        reverse_weight_growth=False,
        start_weight=0.75,
        step_size=10,
        min_point=20,
    )

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=classifier,
        grad_cam=iterative_heatmapping,
        testdataloader=testdataloader,
        num_drops=args.n_drops,
        steps=50,
        num_iterations=args.n_samples,
        update_cam=False,
        show_visualization=show_visualization,
        file_prefix='pn3-IHU-',
        plot_title_prefix_text="EXP1 Point GradCAM IHU PointNet",
        create_png=True,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        print_progress=True,
        use_cuda=args.use_cuda,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)