import import_base_path
import numpy as np
import torch
import random
from helpers.point_dropping_experiment_single import PointDropExperiment
from pointnet.dataset import ShapeNetDataset
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from helpers.args_helper import get_args
import time
from grad_cam.vanilla_gradient import Gradient


if __name__ == '__main__':
    args = get_args('pn2')
    # args.use_cuda = True

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
        #unique_path='./unique-network-select')
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    classifier = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()  # eval not needed at this stage unless grad cam is commented

    # experiment on newer method using gradient before max pooling of the third feature layer

    #grad_cam = GradCamThirdModule(classifier, target_layer_names=["2"], cuda=args.use_cuda,
    #                              counterfactual=False, disable_relu=False, k=3)

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=classifier,
        grad_cam=Gradient(classifier, 'absolute', cuda=args.use_cuda),
        testdataloader=testdataloader,
        num_drops=args.n_drops,
        steps=50,
        num_iterations=args.n_samples,
        update_cam=False,
        show_visualization=False,
        file_prefix='exp13-pn2-gradients-absolute-',
        plot_title_prefix_text="EXP13 Gradients PointNet++",
        create_png=True,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        print_progress=True,
        use_cuda=args.use_cuda,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)

    # grad_cam_wrapper = GradcamWrapper(grad_cam, k=9)
    #
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=args.num_drops,
    #     steps=50,
    #     num_iterations=args.num_iterate,
    #     update_cam=True,
    #     show_visualization=False,
    #     file_prefix='pn2maxpool-newModel_249-k9',
    #     create_png=False,
    #     random_drop=True,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)

    # grad_cam.counterfactual = True
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=2048,
    #     steps=50,
    #     num_iterations=2874,  # max is 2874 = test set size
    #     update_cam=True,
    #     show_visualization=show_visualization,
    #     file_prefix='pn2maxpool-newModel_249-counterfactual-',
    #     alternative_colors=True,
    #     random_drop=True,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)