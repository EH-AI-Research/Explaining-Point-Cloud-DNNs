import import_base_path
import numpy as np
import torch
import random
from pointnet.dataset import ShapeNetDataset
from saliency_maps.saliency_method import GradCamWrapper
from helpers.point_dropping_experiment_single import PointDropExperiment
from helpers.args_helper import get_args
import time
from dgcnn.model import DGCNN


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
        # unique_path='./unique-planes'
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    print("len(test_dataset.classes)", len(test_dataset.classes))

    def get_classifier(num_points_foo):
        classifier = DGCNN(args, output_channels=len(test_dataset.classes))
        if args.use_cuda:
            classifier.cuda()
        classifier.load_state_dict(torch.load(args.model))
        classifier.eval()  # eval not needed at this stage unless grad cam is commented
        return classifier

    num_points = args.num_points
    num_drops = args.n_drops
    grad_cam = GradCamWrapper(get_classifier=get_classifier, num_classes=16, num_drop_points=num_drops,
                              heatmap_updates=False)

    #grad_cam = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda,
    #                   counterfactual=False, disable_relu=False)

    start = time.time()
    drop_experiment = PointDropExperiment(
        classifier=get_classifier(num_points),
        grad_cam=grad_cam,
        testdataloader=testdataloader,
        num_drops=num_drops,
        steps=50,
        steps_heatmap_update=10000,  # ignore, since update_cam is False
        num_iterations=args.n_samples,
        update_cam=False,
        use_cuda=args.use_cuda,
        show_visualization=False,
        create_png=False,
        random_drop=False,
        high_drop=True,
        low_drop=True,
        file_prefix="exp10-dgcnn-saliency-",
        plot_title_prefix_text="EXP10 Saliency DGCNN",
        insertion_experiment=False,
        print_progress=True,
        show_marker=True,
        save_dropping_results=True,
    ).run_experiment()
    print("done in ", time.time() - start)