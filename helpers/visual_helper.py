import open3d as o3d

import sys
from os import path, chdir, listdir
from os.path import isfile, join
base_path = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.append(base_path)
chdir(base_path)
import time
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet.dataset import ShapeNetDataset
from datasets.eh_pointcloud_dataset import EHDataset
import cv2
import numpy as np

def load_testdataset(point_path, num_points=2048, dataset_name='shapenet', use_normals=False,
                     shapenet_unique=False, shapenet_unique_path=''):
    if dataset_name == 'shapenet':
        test_dataset = ShapeNetDataset(
            root=point_path,
            split='test',
            classification=True,
            npoints=num_points,
            data_augmentation=False,
            unique=shapenet_unique,
            unique_path=shapenet_unique_path)
    else:
        filename_filter = [f.split('.')[0] for f in listdir(point_path)]
        test_dataset = EHDataset(
            root="./datasets/eh_mmm",
            config={'n_points': num_points,
                    'models_csv': 'classes-4-vs-8-drills-simple.csv',
                    'filename_filter': filename_filter,
                    'rotate_axes': '',
                    'move_along_axis': '',
                    'data_augmentation_jitter': False,
                    'point_normals': use_normals,
                    },
            split='all'
        )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)  # batch Size set to 1 obtain a single example

    return test_dataset, testdataloader


def visualize_heatmaps(grad_cam, point_path, num_points=2048, num_iteration=100, dataset_name='shapenet',
                       discretize=False, discretization_threshold=0.5, shuffle_input_points=False,
                       use_index_heatmap=False, use_normals=False, save_png_path=None, save_png_prefix='',
                       shapenet_unique=False, shapenet_unique_path='', vis_pre=None, vis_box=None, crop_image=True):
    test_dataset, test_dataloader = load_testdataset(point_path, num_points, dataset_name=dataset_name,
                                                     use_normals=use_normals, shapenet_unique=shapenet_unique,
                                                     shapenet_unique_path=shapenet_unique_path)
    data_iter = iter(test_dataloader)

    if isinstance(grad_cam, list):
        grad_cams = grad_cam
    else:
        grad_cams = [grad_cam]

    if save_png_path:
        # Visualize Point Cloud
        first_done = False
        if not vis_pre:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.create_window(width=960, height=960, visible=True)
            box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
        else:
            vis = vis_pre
            box = vis_box

    i = -1
    for data in data_iter:  # i in range(num_iteration):
        i += 1
        #if len()
        #data = next(data_iter)
        points, target = data

        model = test_dataloader.dataset.last_model
        model_name = model['name']
        print("3D model name:", model_name)

        points, target = Variable(points), Variable(target[:, 0])
        points = points.transpose(2, 1)
        input, target = points.cuda(), target.cuda()

        if shuffle_input_points:
            shuffled_indices = torch.randperm(input.shape[2])
            input = input[:, :, shuffled_indices]

        heatmaps = []
        offset = -int(len(grad_cams) / 2)

        points = points[:, :3, :]  # get rid of normals
        input_original = input
        input = input[:, :3, :]

        if offset <= -1:
            points = input[0].transpose(1, 0).contiguous().data.cpu()
            points = points[:,] + offset - 1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.4, 0.4, 0.4])
            heatmaps.append(pcd)

        for grad_cam in grad_cams:
            #grad_name = None
            #if isinstance(grad_cam, dict):
            #    grad_name = grad_cam['name']
            #    grad_cam = grad_cam['method']

            mask = grad_cam(input_original, target)
            #return mask, shuffled_indices

            if use_index_heatmap:
                # map into a uniformaly distbiuted heatmap based on index sorted by original heatmap values
                num_points = mask.shape[0]
                heatmap_values = np.flip(np.arange(num_points)) / num_points
                heatmap_value_pointer = 0
                max_idx = np.flip(np.argsort(mask))
                for idx in max_idx:
                    mask[idx] = heatmap_values[heatmap_value_pointer]
                    heatmap_value_pointer += 1

            if discretize:
                mask = np.digitize(mask, bins=[discretization_threshold])
                mask = np.clip(mask, 0.15, 0.9)

            colored_re_pointcloud, _ = heatmap_plt(input, mask, offset=offset)

            print('Decision:')
            classes = test_dataset.classes_temp
            soft_pred = F.softmax(grad_cam.classifier_output.data)
            pred_choice = grad_cam.classifier_output.data.max(1)[1]
            idx = pred_choice.cpu().numpy()[0]
            prob = soft_pred[0, idx]
            prob = prob.cpu().numpy()
            pred_class = classes.get(pred_choice.cpu().numpy()[0])

            correct_class = classes.get(target.cpu().numpy()[0])
            pred_is_correct = pred_class == correct_class
            print('correct_class:', correct_class)
            print('pred_class:', pred_class)
            print("correctly predicted?", pred_is_correct)
            print('Confidence: ', prob)
            heatmaps.append(colored_re_pointcloud)
            offset += 1

            if not save_png_path:
                o3d.visualization.draw_geometries(heatmaps)

            if save_png_path:
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # vis.add_geometry(heatmaps[0])
                # vis.poll_events()
                # vis.update_renderer()
                # time_suffix = int(round(time.time() * 1000))
                # vis.capture_screen_image(os.path.join(save_png_path, f"{model_name}-{save_png_prefix}-{time_suffix}.jpg"), True)
                # vis.destroy_window()
                pcd = heatmaps[0]

                # Visualize Point Cloud
                #vis = o3d.visualization.Visualizer()
                #vis.create_window()
                #vis.create_window(window_name=clouds[i], width=960, height=540, visible=True)
                #if prev_pcd:
                #    vis.remove_geometry(prev_pcd)
                vis.add_geometry(pcd)
                #if first_done:
                vis.remove_geometry(box)

                opt = vis.get_render_option()  # needs to be called after create_window has been called
                opt.point_size = 7.0  # default is 5.0
                opt.show_coordinate_frame = False
                opt.background_color = np.asarray([1, 1, 1])

                ctr = vis.get_view_control()
                #ctr.rotate(0, 300)
                #ctr.scale(-6)
                ctr.change_field_of_view(-6.0)
                ctr.set_lookat([0.0, 0.0, 0.0])
                if model_name.startswith('Chair'):  # model_name.startswith('Chair'):
                    ctr.set_front([0.79796494026, 0.5722092, -0.18928])
                    ctr.set_up([-0.5476, 0.8194183, 0.1688534])
                elif model_name.startswith('Table'):  # model_name.startswith('Chair'):
                    ctr.set_front([ 0.59420061615614106, 0.64223908748650127, -0.48420510350900725 ])
                    ctr.set_up([ -0.51653621959334339, 0.76616160244616904, 0.38234373642749264 ])
                    ctr.set_lookat([ -0.0051378160715103149, -0.27331794798374176, 0.00019463896751403809 ])
                elif model_name.startswith('Airplane'):
                    ctr.set_front([0.51463953400160645, 0.80399770941192172, -0.29788224737804153])
                    ctr.set_up([-0.64408693772999981, 0.59184060880958866, 0.4846408055555031])
                    ctr.set_lookat([-0.0051378160715103149, -0.27331794798374176, 0.00019463896751403809])
                elif model_name.startswith('Lamp'):
                    ctr.set_front([0.63225730823217363, 0.091875904698941663, 0.76929156652257491])
                    ctr.set_up([-0.079822379645474367, 0.99538446013647497, -0.053274423756170641])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                elif model_name.startswith('Motorbike'):
                    ctr.set_front([0.49372662876406509, 0.34597363310199369, 0.79783222625278705])
                    ctr.set_up([-0.16795824045239735, 0.93811567414174424, -0.30286797683762379])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                elif model_name.startswith('Skateboard'):
                    ctr.set_front([-0.54297061720396467, 0.58517443496033383, -0.60229045279001381])
                    ctr.set_up([0.49066626961995818, 0.80313340536523758, 0.3379694439497033])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                else:
                    ctr.set_front([0.0, 1.0, 0.0])
                    ctr.set_up([0.0, 0.0, 1.0])

                # Updates
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                # Capture image
                time.sleep(0.5)
                time_suffix = int(round(time.time() * 1000))
                vis.run()
                pred_is_correct_label = 'correct' if pred_is_correct else 'false'
                prefix = save_png_prefix
                #if grad_name:
                #    prefix = save_png_prefix + grad_name
                img_path = os.path.join(save_png_path, f"{model_name}-{prefix}-{pred_is_correct_label}-{time_suffix}.png")
                vis.capture_screen_image(img_path, do_render=True)
                vis.add_geometry(box)
                vis.remove_geometry(pcd)

                if crop_image:
                    crop_image_white_background(img_path)
                #first_done = True
                #vis.update_renderer()

                #time.sleep(1)
                #time.sleep(1)
                #del ctr
                #del vis

                #prev_pcd = pcd
    #vis.destroy_window()


def heatmap_plt(input, mask, offset=0, colormap=True):
    if colormap is True:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    else:
        heatmap = mask
    heatmap = np.float32(heatmap) / 255
    try:
        heatmap = np.squeeze(heatmap, axis=1)
        # heatmap = np.squeeze(heatmap, axis=1)
        heatmap = np.squeeze(heatmap)  # , axis=1
        heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]
    except:
        pass
    pcd_list = []
    pcd_list_orig = []
    for i in range(int(input.size(2))):
        pcd_ = o3d.geometry.PointCloud()
        pcd_list.append(pcd_)
        pcd_list_orig.append(pcd_)

    prc_r_all = input[0].transpose(1, 0).contiguous().data.cpu()

    colored_re_pointcloud2 = o3d.geometry.PointCloud()
    colored_re_pointcloud2_orig = o3d.geometry.PointCloud()

    for j in range(int(input.size(2))):
        current_patch = prc_r_all[j,] + offset
        current_patch = current_patch.unsqueeze(dim=0)
        pcd_list[j].points = o3d.utility.Vector3dVector(current_patch)
        pcd_list[j].paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
        colored_re_pointcloud2 += pcd_list[j]

        current_patch_orig = prc_r_all[j,]
        current_patch_orig = current_patch_orig.unsqueeze(dim=0)
        pcd_list_orig[j].points = o3d.utility.Vector3dVector(current_patch_orig)
        pcd_list_orig[j].paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
        colored_re_pointcloud2_orig += pcd_list_orig[j]

    return colored_re_pointcloud2, colored_re_pointcloud2_orig


def crop_image_white_background(path):
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.bitwise_not(gray)  # invert so that background get black and can be detected by bounrdingRect
    #blur = cv2.GaussianBlur(gray, (3, 3), 0)   # would add a small padding
    #thresh = cv2.threshold(blur, 255, 0, cv2.THRESH_BINARY)[1]
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Obtain bounding rectangle and extract ROI
    x, y, w, h = cv2.boundingRect(thresh)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y:y + h, x:x + w]

    cv2.imwrite(path, ROI)

# def heatmap_plt(input, mask,offset=0):
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     #heatmap = np.squeeze(heatmap, axis=1)
#     heatmap = np.squeeze(heatmap)  # , axis=1
#     heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]
#
#     pcd_list = []
#     for i in range(int(input.size(2))):
#         pcd_ = o3d.geometry.PointCloud()
#         pcd_list.append(pcd_)
#
#     prc_r_all = input[0].transpose(1, 0).contiguous().data.cpu()
#
#     colored_re_pointcloud2 = o3d.geometry.PointCloud()
#     for j in range(int(input.size(2))):
#         current_patch = prc_r_all[j,] + offset
#         current_patch = current_patch.unsqueeze(dim=0)
#         pcd_list[j].points = o3d.utility.Vector3dVector(current_patch)
#         pcd_list[j].paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
#         colored_re_pointcloud2 += pcd_list[j]
#     return colored_re_pointcloud2
#



def point_plt(input, set_size=64, offset=0, paint_gray=False):
    # Visualzie the point cloud first
    hight_light_caps = [np.random.randint(0, set_size) for r in range(int(10))]
    colors = plt.cm.tab20((np.arange(20)).astype(int))
    pcd_list = []
    for i in range(set_size):
        pcd_ = o3d.geometry.PointCloud()
        pcd_list.append(pcd_)

    prc_r_all = input[0].transpose(1, 0).contiguous().data.cpu()
    # print('Shape of prc_r_all : ', np.shape(prc_r_all))
    prc_r_all_point = o3d.geometry.PointCloud()
    prc_r_all_point.points = o3d.utility.Vector3dVector(prc_r_all)
    # print('Shape of prc_r_all Points : ', np.shape(prc_r_all_point.points))

    colored_re_pointcloud1 = o3d.geometry.PointCloud()
    jc = 0
    for j in range(set_size):
        current_patch = torch.zeros(int(input.size(2) / set_size), 3)

        # print('Shape of current_patch : ', np.shape(current_patch))

        for m in range(int(input.size(2) / set_size)):
            current_patch[m,] = prc_r_all[
                                    set_size * m + j,] + offset  # the reconstructed patch of the capsule m is not saved continuesly in the output reconstruction.
        # print('Shape of current_patch : ', np.shape(current_patch))
        pcd_list[j].points = o3d.utility.Vector3dVector(current_patch)
        if paint_gray is False:
            if (j in hight_light_caps):
                pcd_list[j].paint_uniform_color([colors[jc, 0], colors[jc, 1], colors[jc, 2]])
                jc += 1
            else:
                pcd_list[j].paint_uniform_color([0.8, 0.8, 0.8])
        else:
            pcd_list[j].paint_uniform_color([0.8, 0.8, 0.8])
        colored_re_pointcloud1 += pcd_list[j]
    return colored_re_pointcloud1
    # o3d.visualization.draw_geometries([colored_re_pointcloud1])

def heatmap_upscale(inputs,colored_re_pointcloud,offset = 0):
    pcd_tree = o3d.geometry.KDTreeFlann(colored_re_pointcloud)
    seed_points = np.asarray(colored_re_pointcloud.points)
    #seed_points = colored_re_pointcloud.points

    cam_input = np.zeros(len(inputs))

    for i in range(len(inputs)):
        pt = inputs[i]
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 2)

        # apply rervse distance weighted interpolation
        # http://www.gitta.info/ContiSpatVar/de/html/Interpolatio_learningObject2.html
        weighted_cam_sum = 0
        weight_sum = 0
        for id in idx:
            d = np.linalg.norm(seed_points[id] - pt)
            weighted_cam_sum += (1.0 / d) * cam_f[id]
            weight_sum += (1.0 / d)

        cam_input[i] = weighted_cam_sum / weight_sum

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_input), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # heatmap = np.squeeze(heatmap, axis=1)
    heatmap = np.squeeze(heatmap)
    heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]  # only one line Change

    colored_input_pc = o3d.geometry.PointCloud()
    for j in range(len(inputs)):
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector([list(inputs[j]+ offset)])
        pcd_.paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
        colored_input_pc += pcd_
    return colored_input_pc


def heatmap_upscale_method2(inputs, colored_re_pointcloud, neighbours=256,offset = 0):
    colored_input_pc = inputs
    prc_r_all = colored_input_pc[0].contiguous().data.cpu() + offset
    prc_r_all_point = o3d.geometry.PointCloud()
    prc_r_all_point.points = o3d.utility.Vector3dVector(prc_r_all)

    prc_r_orig = colored_input_pc[0].contiguous().data.cpu()
    prc_r_orig_point = o3d.geometry.PointCloud()
    prc_r_orig_point.points = o3d.utility.Vector3dVector(prc_r_orig)


    colored_input_pc = prc_r_all_point
    colored_input_pc.paint_uniform_color([0.8, 0.8, 0.8])
    pcd_tree = o3d.geometry.KDTreeFlann(prc_r_orig_point)

    print("Find its 32 nearest neighbors, paint again.")
    nn = neighbours
    num_points = len(colored_re_pointcloud.points)
    for i in range(0, len(colored_re_pointcloud.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(colored_re_pointcloud.points[i], nn)
        np.asarray(colored_input_pc.colors)[idx[1:], :] = colored_re_pointcloud.colors[i]  # [0, 0, 1]
    return colored_input_pc
