import import_base_path
import numpy as np
import torch
import random
from helpers.point_dropping_experiment_single import PointDropExperiment
from pointnet.dataset import ShapeNetDataset
from dgcnn.model import DGCNN
import time
from grad_cam.grad_cam_dgcnn import GradCamDGCNN
from helpers.args_helper import get_args

if __name__ == '__main__':
    args = get_args('dgcnn')

    print("starting point dropping experiment")

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
        #unique_path='./unique-network-select'
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    #classifier = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
    classifier = DGCNN(args, output_channels=len(test_dataset.classes))
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()  # eval not needed at this stage unless grad cam is commented

    grad_cam = GradCamDGCNN(args, model=classifier, disable_relu=False, use_cuda=args.use_cuda)

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=classifier,
        grad_cam=grad_cam,
        testdataloader=testdataloader,
        num_drops=args.n_drops,
        steps=50,
        num_iterations=args.n_samples,
        update_cam=False,
        show_visualization=False,
        file_prefix='exp6-dgcnn-L1-',
        plot_title_prefix_text="EXP6 Point GradCAM L1 DGCNN",
        create_png=True,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        print_progress=True,
        use_cuda=args.use_cuda,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)

    # counterfactual
    #
    # grad_cam_wrapper.grad_cam.counterfactual = True
    #
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam,
    #     testdataloader=testdataloader,
    #     num_drops=2048,
    #     steps=50,
    #     num_iterations=2874,
    #     update_cam=False,
    #     show_visualization=show_visualization,
    #     file_prefix='dgcnn-L1-counterfactual-',
    #     create_png=True,
    #     random_drop=False,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    # ).run_experiment()