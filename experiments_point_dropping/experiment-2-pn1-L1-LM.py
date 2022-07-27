import import_base_path
import numpy as np
import torch
import random
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
from grad_cam.grad_cam_pointnet1 import GradCam
from helpers.point_dropping_experiment_single import PointDropExperiment
from helpers.args_helper import get_args
import time

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
        # unique_path='./unique-planes'
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    print("len(test_dataset.classes)", len(test_dataset.classes))
    classifier = PointNetCls(k=len(test_dataset.classes), num_points=args.num_points)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))

    grad_cam = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda,
                       counterfactual=False, disable_relu=False)

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=classifier,
        grad_cam=grad_cam,
        testdataloader=testdataloader,
        num_drops=args.n_drops,
        steps=50,
        steps_heatmap_update=50,
        num_iterations=args.n_samples,
        update_cam=True,
        show_visualization=False,
        create_png=False,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        file_prefix="exp2-pn1-L1_to_LM-",
        plot_title_prefix_text="EXP1 Point GradCAM L1-LM PointNet",
        insertion_experiment=False,
        print_progress=True,
        show_marker=True,
        use_cuda=args.use_cuda,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)