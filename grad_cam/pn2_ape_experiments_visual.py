
if __name__ == '__main__':
    args = get_args()

    torch.backends.cudnn.deterministic = True
    random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    test_dataset = ShapeNetDataset(
        root=args.point_path,
        split='test',
        classification=True,
        npoints=args.num_points,
        data_augmentation=False,
        unique=True,
        unique_path='./unique-network-comparison-debug')

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True)  # batch Size set to 1 obtain a single example

    test_iter = iter(testdataloader)

    for i in range(100):
        data = next(test_iter)  # get a single batch from the dataloader
        points, targets = data
        points, targets = Variable(points), Variable(targets[:, 0])
        points = points.transpose(2, 1)
        input, target = points.cuda(), targets.cuda()
        input = Variable(input, requires_grad=True)

        colored_re_pointcloud = point_plt(input)

        # input_np1_t = input.data.cpu().numpy()[0].T
        # double = 0
        # already = []
        # for ip1 in range(input_np1_t.shape[0]):
        #     id = f"{input_np1_t[ip1][0]}-{input_np1_t[ip1][1]}-{input_np1_t[ip1][2]}"
        #     if id in already:
        #         continue
        #
        #     for ip2 in range(input_np1_t.shape[0]):
        #         if ip1 == ip2:
        #             continue
        #         if input_np1_t[ip1][0] == input_np1_t[ip2][0] and input_np1_t[ip1][1] == input_np1_t[ip2][1] and input_np1_t[ip1][2] == input_np1_t[ip2][2]:
        #             if id not in already:
        #                 already.append(id)
        #             print("gleich", input_np1_t[ip1], input_np1_t[ip2])
        #             double = double + 1
        #
        # input_unique_count = np.unique(input_np1_t, axis=0)
        # print("targets", targets)

        # 1454 unique


        #TODO New code added for progressive point dropping algorithm, please ignore the old implementation of
        # point dropping algorithim ***progress_mode*** , the correct one is ****progress_mode_input***

        # Grad Cam for pointnet by INPUT progressive point dropping
        net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        net.eval()  # eval not needed at this stage unless grad cam is commented

        grad_cam_progress = GradCamPointnet2(args, module_n='1',
                                             interp_mode=False,
                                             progress_mode_input=True,
                                             num_point_drops=2048,
                                             progress_mode_input_plus_interpolate=False,
                                             progress_mode_input_plus_interpolate_k=3,
                                             progress_mode_input_old_droping_indecing_loop=True)
        grad_cam_progress(input, net, target.cpu().data.numpy())
        cam_f_list = grad_cam_progress.cam_progress
        xyz_list = grad_cam_progress.grad_xyz_progress
        # input_xyz_list = grad_cam_progress.input_xyz_new
        # Aggregate CAM
        # xyz_aggregate = xyz_list[0]
        # cam_aggregate = cam_f_list[0]
        # for i in range(1, len(xyz_list)):
        #     xyz_aggregate = torch.cat((xyz_aggregate, xyz_list[i]), 1)
        #     cam_aggregate = np.concatenate((cam_aggregate, cam_f_list[i]), 0)
        #
        # colored_re_pointcloud_aggregate, colored_re_pointcloud_offset_aggregate = heatmap_plt(
        #     xyz_aggregate.transpose(1, 2), cam_aggregate, offset=8)

        #####
        ### interpolation
        ###
        # interpolate = True
        # if interpolate:
        #     pcd_tree = o3d.geometry.KDTreeFlann(colored_re_pointcloud_offset_aggregate)
        #     # seed_points = np.asarray(colored_re_pointcloud_aggregate.points)
        #
        #     # cam_i = np.zeros(len(inputs))
        #     input_np_flipped = input.data.cpu().numpy()[0].T
        #     cam_input = np.zeros(input_np_flipped.shape[0])
        #     for i in range(input_np_flipped.shape[0]):
        #         pt = input_np_flipped[i]
        #         [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 8)
        #
        #         # apply rervse distance weighted interpolation
        #         # http://www.gitta.info/ContiSpatVar/de/html/Interpolatio_learningObject2.html
        #         weighted_cam_sum = 0
        #         weight_sum = 0
        #         for id in idx:
        #             d = max(0.1, np.linalg.norm(input_np_flipped[id] - pt)) # min distance 0.1
        #             weighted_cam_sum += (1.0 / d) * cam_aggregate[id]
        #             weight_sum += (1.0 / d)
        #
        #         cam_input[i] = weighted_cam_sum / weight_sum
        #
        #     colored_re_pointcloud_aggregate, colored_re_pointcloud_offset_aggregate_interpolated = heatmap_plt(
        #         input, cam_input, offset=8)
        ### interpolation end
        #####

        xyz_npa = grad_cam_progress.xyz_aggregate.data.cpu().numpy()

        # double_counter = 0
        # for x in xyz_npa[0]:
        #     next = False
        #     for y in xyz_npa[0]:
        #         if next:
        #             break
        #         if x > y and x[0] == y[0] and x[1] == y[1] and x[2] == y[2]:
        #             next = True
        #         if x < y and x[0] == y[0] and x[1] == y[1] and x[2] == y[2]:
        #             double_counter = double_counter + 1
        # print("double_counter ", double_counter)

        colored_re_pointcloud_aggregate, colored_re_pointcloud_offset_aggregate = heatmap_plt(
            grad_cam_progress.xyz_aggregate.transpose(1, 2), grad_cam_progress.cam_aggregate, offset=8)


        xyz_unique = np.unique(xyz_npa, axis=1) # should be 1454 unique


        colored_re_pointcloud_list = []
        colored_re_pointcloud_list_offset = []

        for i in range(len(xyz_list)):
            x,y = heatmap_plt(xyz_list[i].transpose(1,2), cam_f_list[i],offset=int(i*3))
            colored_re_pointcloud_list.append(y)
            colored_re_pointcloud_list_offset.append(x)


        #progess_xyz_0 = point_plt(torch.Tensor(grad_cam_progress.input_xyz_new[0]).transpose(1,2))
        #progess_xyz_1 = point_plt(torch.Tensor(grad_cam_progress.input_xyz_new[1]).transpose(1,2),offset=2)
        #progess_xyz_2 = point_plt(torch.Tensor(grad_cam_progress.input_xyz_new[2]).transpose(1,2),offset=4)
        #progess_xyz_3 = point_plt(torch.Tensor(grad_cam_progress.input_xyz_new[3]).transpose(1,2),offset=6)

        #o3d.visualization.draw_geometries([progess_xyz_0,progess_xyz_1,progess_xyz_2,progess_xyz_3])#colored_input_pc_upscale1,
        #o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[0],colored_re_pointcloud_list_offset[1],colored_re_pointcloud_list_offset[2],colored_re_pointcloud_list_offset[3],colored_re_pointcloud_offset_aggregate])#colored_input_pc_upscale1,
        #o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[15]])
        #o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[0]])#colored_input_pc_upscale1,



        o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[0], colored_re_pointcloud_list_offset[1],
                                           colored_re_pointcloud_list_offset[2], colored_re_pointcloud_list_offset[3]])
        o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[4], colored_re_pointcloud_list_offset[5],
                                           colored_re_pointcloud_list_offset[6], colored_re_pointcloud_list_offset[7]])
        o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[8], colored_re_pointcloud_list_offset[9],
                                           colored_re_pointcloud_list_offset[10], colored_re_pointcloud_list_offset[11]])
        o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[12], colored_re_pointcloud_list_offset[13],
                                           colored_re_pointcloud_list_offset[14]])
        o3d.visualization.draw_geometries([colored_re_pointcloud_offset_aggregate])#colored_input_pc_upscale1,
        # o3d.visualization.draw_geometries([colored_re_pointcloud_offset_aggregate_interpolated])#colored_input_pc_upscale1,

















        # Grad Cam for pointnet by Accumulation method
        off = 1
        net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        net.eval()  # eval not needed at this stage unless grad cam is commented
        grad_cam = GradCamPointnet2(args, module_n='1',interp_mode=False,progress_mode=False)
        cam_f = grad_cam(input, net)
        colored_re_pointcloud = point_plt(input)
        colored_re_pointcloud2,_ = heatmap_plt(grad_cam.features_old_xyz[0], cam_f, offset=off)
        off +=1
        # Grad Cam for pointnet by interpolation method

        net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        net.eval()  # eval not needed at this stage unless grad cam is commented
        grad_cam = GradCamPointnet2(args, module_n='1',interp_mode=True,progress_mode=False)
        cam_f = grad_cam(input, net)
        colored_re_pointcloud2_interp,colored_re_pointcloud2_interp_orig = heatmap_plt(grad_cam.features_xyz.transpose(1,2), cam_f, offset=off)
        off += 1
        colored_re_pointcloud2_interp_upscale = heatmap_upscale(input.squeeze().transpose(0,1).data.cpu().numpy(),colored_re_pointcloud2_interp_orig,offset = off)
        off += 1
        colored_re_pointcloud2_interp_upscale2 = heatmap_upscale_method2(input.transpose(2,1),colored_re_pointcloud2_interp_orig, offset= off)
        off += 1

        # Grad Cam for pointnet by progressive point dropping
        net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        net.eval()  # eval not needed at this stage unless grad cam is commented
        grad_cam_progress = GradCamPointnet2(args, module_n='1',interp_mode=False,progress_mode=True)
        cam_f = grad_cam_progress(input, net)
        cam_f_list = grad_cam_progress.cam_progress
        xyz_list = grad_cam_progress.features_xyz


        colored_re_pointcloud_list = []
        colored_re_pointcloud_list_offset = []

        for i in range(len(xyz_list)):
            x,y = heatmap_plt(xyz_list[i].transpose(1,2), cam_f_list[i],offset =int(i*3))
            colored_re_pointcloud_list.append(y)
            colored_re_pointcloud_list_offset.append(x)


        progess_xyz_0 = point_plt(torch.Tensor(grad_cam_progress.seed_xyz_new[0]).transpose(1,2))
        progess_xyz_1 = point_plt(torch.Tensor(grad_cam_progress.seed_xyz_new[1]).transpose(1,2),offset=2)
        progess_xyz_2 = point_plt(torch.Tensor(grad_cam_progress.seed_xyz_new[2]).transpose(1,2),offset=4)
        progess_xyz_3 = point_plt(torch.Tensor(grad_cam_progress.seed_xyz_new[3]).transpose(1,2),offset=6)


        o3d.visualization.draw_geometries([progess_xyz_0,progess_xyz_1,progess_xyz_2,progess_xyz_3])#colored_input_pc_upscale1,
        o3d.visualization.draw_geometries([colored_re_pointcloud_list_offset[0],colored_re_pointcloud_list_offset[1],colored_re_pointcloud_list_offset[2],colored_re_pointcloud_list_offset[3]])#colored_input_pc_upscale1,







        net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)

        net.cuda()
        net.load_state_dict(torch.load(args.model))
        net.eval()

        print('Pointnet 2 Decisions:')
        classIS = test_dataset.classes_temp

        with torch.no_grad():
            pred = net(input)
            soft_pred = F.softmax(pred)
            pred_choice = pred.data.max(1)[1]
            idx = pred_choice.cpu().numpy()[0]
            prob = soft_pred[0, idx]
            prob = prob.cpu().numpy()
            pred_class = classIS.get(pred_choice.cpu().numpy()[0])

            # correct = pred_choice.eq(target.data).cpu().sum()

            correct_class = classIS.get(targets.cpu().numpy()[0])
            print('correct_class:', correct_class)
            print('pred_class:', pred_class)
            print('Confidence: ', prob)

        del net,grad_cam
        torch.cuda.empty_cache()

        classifier = PointNetCls(k=len(test_dataset.classes), num_points=args.num_points)
        classifier.cuda()
        classifier.load_state_dict(torch.load(args.model_p1))
        grad_cam = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda)

        target_index = None
        mask = grad_cam(input, target_index)

        colored_re_pointcloud1,_ = heatmap_plt(input, mask, offset=off)
        off += 1

        print('Pointnet 1 Decisions:')
        classIS = test_dataset.classes_temp
        with torch.no_grad():
            pred = classifier(input)
            soft_pred = F.softmax(pred)
            pred_choice = pred.data.max(1)[1]
            idx = pred_choice.cpu().numpy()[0]
            prob = soft_pred[0, idx]
            prob = prob.cpu().numpy()
            pred_class = classIS.get(pred_choice.cpu().numpy()[0])

            # correct = pred_choice.eq(target.data).cpu().sum()

            correct_class = classIS.get(targets.cpu().numpy()[0])
            print('correct_class:', correct_class)
            print('pred_class:', pred_class)
            print('Confidence: ', prob)

        classifier = PointNetCls(k=len(test_dataset.classes), num_points=args.num_points)
        classifier.cuda()
        classifier.load_state_dict(torch.load(args.model_p1))
        grad_cam = GradCam(model=classifier, target_layer_names=["7"], use_cuda=args.use_cuda)

        target_index = None
        mask = grad_cam(input, target_index)

        colored_re_pointcloud1, _ = heatmap_plt(input, mask, offset=off)
        off += 1

        print('Pointnet 1 Decisions:')
        classIS = test_dataset.classes_temp
        with torch.no_grad():
            pred = classifier(input)
            soft_pred = F.softmax(pred)
            pred_choice = pred.data.max(1)[1]
            idx = pred_choice.cpu().numpy()[0]
            prob = soft_pred[0, idx]
            prob = prob.cpu().numpy()
            pred_class = classIS.get(pred_choice.cpu().numpy()[0])

            # correct = pred_choice.eq(target.data).cpu().sum()

            correct_class = classIS.get(targets.cpu().numpy()[0])
            print('correct_class:', correct_class)
            print('pred_class:', pred_class)
            print('Confidence: ', prob)

        print('Input','PointNet 2 Accumulation','PointNet 2 Interpolation','PointNet 2 Interp Upscale Method 1','PointNet 2 Interp Upscale Method 2','PointNet 1')




        o3d.visualization.draw_geometries(
            [colored_re_pointcloud, colored_re_pointcloud2,colored_re_pointcloud2_interp,colored_re_pointcloud2_interp_upscale,colored_re_pointcloud2_interp_upscale2,colored_re_pointcloud1])  # , colored_re_pointcloud1, colored_re_pointcloud2

        del classifier, grad_cam
        torch.cuda.empty_cache()
        print('done')
