import open3d as o3d
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
import os

class PointDropExperiment:

    def __init__(
            self,
            classifier,
            grad_cam,
            testdataloader,
            num_drops=1024,
            show_visualization=False,
            file_prefix="",
            steps=250,
            steps_heatmap_update=False,  # defaults to steps
            num_iterations=10,
            update_cam=True,
            alternative_colors=False,
            show_marker=True,
            create_png=False,
            random_drop=True,
            random_drop_prior_mask=False,
            high_drop=True,
            low_drop=True,
            print_progress=False,
            use_cam_points=False,
            diagram_min_y=False,
            diagram_max_y=False,
            plt_max_x=False,
            insertion_experiment=False,
            create_confidence_plot=True,
            create_loss_plot=True,
            plot_guess_line=True,
            plot_title=True,
            plot_title_prefix_text='',
            use_cuda=True,
            save_dropping_results=False,
            update_heatmap_for_low_dropping=False):
        self.use_cuda = use_cuda
        self.steps = steps
        if steps_heatmap_update is False:
            self.steps_heatmap_update = steps
        else:
            self.steps_heatmap_update = steps_heatmap_update
        self.classifier = classifier
        self.grad_cam = grad_cam
        self.testdataloader = testdataloader
        self.show_marker = show_marker
        self.insertion_experiment = insertion_experiment

        self.show_visualization = show_visualization
        self.diagram_min_y = diagram_min_y
        self.diagram_max_y = diagram_max_y
        self.plt_max_x = plt_max_x
        self.num_drops = num_drops
        self.num_classified = 0
        self.num_iterations = num_iterations
        self.alternative_colors = alternative_colors

        self.create_confidence_plot = create_confidence_plot
        self.create_loss_plot = create_loss_plot
        self.plot_guess_line = plot_guess_line
        self.plot_title = plot_title
        self.plot_title_prefix_text = plot_title_prefix_text

        #num_data_points = num_drops + 1
        num_data_points = 1 # for 0 drop values
        for i in range(self.num_drops):
            if self.measure_condition(i + 1):
                num_data_points += 1

        self.random_correct = np.zeros(num_data_points)
        self.random_target_confidence = np.zeros(num_data_points)
        self.random_loss = np.zeros(num_data_points)

        self.high_correct = np.zeros(num_data_points)
        self.high_target_confidence = np.zeros(num_data_points)
        self.high_loss = np.zeros(num_data_points)

        self.low_correct = np.zeros(num_data_points)
        self.low_target_confidence = np.zeros(num_data_points)
        self.low_loss = np.zeros(num_data_points)

        self.file_prefix = file_prefix

        # create point cloud and reuse it
        self.pcd = o3d.geometry.PointCloud()
        self.update_cam = update_cam

        self.create_png = create_png

        self.random_drop = random_drop
        self.high_drop = high_drop
        self.low_drop = low_drop
        self.random_drop_prior_mask = random_drop_prior_mask

        self.print_progress = print_progress

        self.use_cam_points = use_cam_points

        self.iteration_index = 0

        self.foo = 1
        self.save_dropping_results = save_dropping_results

        self.update_heatmap_for_low_dropping = update_heatmap_for_low_dropping

    def run_experiment(self):
        testdata_iter = iter(self.testdataloader)
        for i in range(self.num_iterations):
            self.iteration_index = i
            start = time.time()
            data = next(testdata_iter)  # get a single batch from the dataloader
            points, target = data
            points, target = Variable(points), Variable(target[:, 0])
            points = points.transpose(2, 1)
            input, target = points, target
            if self.use_cuda:
                input, target = input.cuda(), target.cuda()

            self.drop_experiment(input, target)

            dur = time.time() - start
            if self.print_progress:
                print('experiment:', self.num_iterations, 'pcds', self.num_drops, 'drops', self.steps, 'steps')
                print(i, 'done in', dur, 'seconds')
                print('estimated time untill complete:', (self.num_iterations - i) * dur, 'seconds')

        self.save_results()
        self.plot_results()

    def drop_experiment(
            self,
            input,
            target_idx,
            ignore_false=False):
        #target = torch.zeros((1, num_output_classes), dtype=torch.long).cuda()
        #target[0, target_idx] = 1
        target = torch.tensor([target_idx], dtype=torch.long)
        if self.use_cuda:
            target = target.cuda()

        if self.insertion_experiment:
            centered_input = input.clone().detach()
            centered_input[:, :, :] = 0

        cam = self.grad_cam(input, target_idx)

        if self.use_cam_points: # p++ ppd experiment
            input = self.grad_cam.points

        #print("cam size", cam.shape)
        self.update_heatmap(input, cam)
        self.show_pcd(input)

        if self.insertion_experiment:
            initial_result = self.classifier(centered_input)
            initial_class_id = initial_result.data.argmax(1)[0]
        else:
            initial_result = self.grad_cam.classifier_output
            initial_class_id = initial_result.data.argmax(1)[0]

        # add 0 drop confidence values
        initial_confidence = self.get_confidence(initial_result, target_idx)
        self.random_target_confidence[0] += initial_confidence
        self.high_target_confidence[0] += initial_confidence
        self.low_target_confidence[0] += initial_confidence

        # add 0 drop loss values
        initial_loss = self.get_loss(initial_result[0], target)
        self.random_loss[0] += initial_loss
        self.high_loss[0] += initial_loss
        self.low_loss[0] += initial_loss

        # add 0 drop accuracy values
        initial_correct = 1 if initial_class_id == target_idx[0] else 0
        self.random_correct[0] += initial_correct
        self.high_correct[0] += initial_correct
        self.low_correct[0] += initial_correct

        if self.random_drop:
            # copy input
            points = centered_input.clone().detach() if self.insertion_experiment else input.clone().detach()

            # get self.num_drops random integer numbers between 0 and points.size(2)
            rand_idx = np.random.choice(points.size(2), size=self.num_drops, replace=False)
            #if self.random_drop_prior_mask: # just an experiment...
            #    rand_mask = np.random.choice(point.size(2))
            j = 1

            for i in range(self.num_drops):
                idx = rand_idx[i]
                if self.insertion_experiment:
                    points[:, :, idx] = input[:, :, idx] # shift point to original position
                else:
                    points[:, :, idx] = 0 # shift point to center

                if self.measure_condition(i + 1):
                    # measure metrics
                    result = self.classifier(points)
                    #result = self.grad_cam.only_classify(points)
                    class_id = result.data.argmax(1)[0]
                    self.random_target_confidence[j] += self.get_confidence(result, target_idx)
                    self.random_loss[j] += self.get_loss(result, target)
                    self.random_correct[j] += 1 if class_id == target_idx[0] else 0
                    j += 1
                    # del result

            self.show_pcd(points)
            del points

        if self.high_drop:
            # copy input
            points = centered_input.clone().detach() if self.insertion_experiment else input.clone().detach()
            hcam = np.copy(cam)
            # hcam_gate is necessary to ignore the cam values of the already ignored points
            hcam_add_gate = np.zeros(hcam.size, dtype=np.dtype(np.float32))
            k = 1

            for i in range(self.num_drops):
                idx = np.argmax(hcam)
                # set cam value to zero so that the next argmax call the return the next lower value
                hcam[idx] = -1.5
                hcam_add_gate[idx] = -2.5
                if self.insertion_experiment:
                    points[:, :, idx] = input[:, :, idx] # shift point to original position
                else:
                    points[:, :, idx] = 0 # shift point to center
                result = None

                if self.heatmap_update_condition(i + 1):
                    # update cam
                    hcam = self.grad_cam(points, target_idx)
                    if np.isnan(hcam).any():
                        hcam = np.zeros(hcam.size)
                    self.update_heatmap(points, hcam)
                    hcam = hcam + hcam_add_gate
                    result = self.grad_cam.classifier_output

                if self.measure_condition(i + 1):
                    # measure metrics
                    if result is None:
                        result = self.classifier(points)
                    class_id = result.data.argmax(1)[0]
                    self.high_target_confidence[k] += self.get_confidence(result, target_idx)
                    self.high_loss[k] += self.get_loss(result, target)
                    self.high_correct[k] += 1 if class_id == target_idx[0] else 0
                    k = k + 1

            self.show_pcd(points)

        if self.update_heatmap_for_low_dropping:
            cam = self.grad_cam(input, target_idx, low_drop=True)

        if self.low_drop:
            # copy input
            points = centered_input.clone().detach() if self.insertion_experiment else input.clone().detach()
            lcam = np.copy(cam)
            # lcam_gate is necessary to ignore the cam values of the already ignored points
            lcam_add_gate = np.zeros(lcam.size)
            j = 1

            for i in range(self.num_drops):
                idx = np.argmin(lcam)
                # set cam value to one so that the next argmin call the return the next higher value
                lcam[idx] = 1.5
                lcam_add_gate[idx] = 1.5
                if self.insertion_experiment:
                    points[:, :, idx] = input[:, :, idx]  # shift point to original position
                else:
                    points[:, :, idx] = 0  # shift point to center
                result = None

                if self.heatmap_update_condition(i + 1):
                    # update cam
                    lcam = self.grad_cam(points, target_idx)
                    if np.isnan(lcam).any():
                        lcam = np.zeros(lcam.size)
                    self.update_heatmap(points, lcam)
                    lcam = lcam + lcam_add_gate
                    result = self.grad_cam.classifier_output

                if self.measure_condition(i + 1):
                    # measure metrics
                    if result is None:
                        result = self.classifier(points)
                    class_id = result.data.argmax(1)[0]
                    self.low_target_confidence[j] += self.get_confidence(result, target_idx)
                    self.low_loss[j] += self.get_loss(result, target)
                    self.low_correct[j] += 1 if class_id == target_idx[0] else 0
                    j += 1

                if i == 512:
                    self.show_pcd(points)

                if i == 1024:
                    self.show_pcd(points)


            self.show_pcd(points)

        self.num_classified += 1

        #print("Done")

    def measure_condition(self, i):
        # in the first and last 50 drops, force a step of 5
        #return i % self.steps == 0 or (i < 50 and i % 5 == 0) or (i > self.num_drops - 50 and i % 5 == 0)
        #return i % self.steps == 0 or (i > (self.num_drops - 47) and (i - 2000) % 24 == 0)
        return i % self.steps == 0 or i == self.num_drops - 1

    def heatmap_update_condition(self, i):
        return self.update_cam and i % self.steps_heatmap_update == 0

    def save_results(self):
        if not self.save_dropping_results:
            return
        np.savez(self.get_file_name(),
                 num_drops=np.array([self.num_drops]),
                 num_classified=np.array([self.num_classified]),
                 random_correct=self.random_correct,
                 random_target_confidence=self.random_target_confidence,
                 random_loss=self.random_loss,
                 high_correct=self.high_correct,
                 high_target_confidence=self.high_target_confidence,
                 high_loss=self.high_loss,
                 low_correct=self.low_correct,
                 low_target_confidence=self.low_target_confidence,
                 low_loss=self.low_loss)

    def get_file_name(self):
        suffix = '-hm_updated' if self.update_cam else ''
        return f"./tmp/mat-{self.file_prefix}{self.num_iterations}pcds-{self.num_drops}drops-{self.steps}steps{suffix}.npz"

    def load_results(self):
        mat = np.load(self.get_file_name())
        self.num_drops = mat['num_drops'][0]
        self.num_classified = mat['num_classified'][0]
        self.random_correct = mat['random_correct']
        self.random_target_confidence = mat['random_target_confidence']
        self.random_loss = mat['random_loss']
        self.high_correct = mat['high_correct']
        self.high_target_confidence = mat['high_target_confidence']
        self.high_loss = mat['high_loss']
        self.low_correct = mat['low_correct']
        self.low_target_confidence = mat['low_target_confidence']
        self.low_loss = mat['low_loss']

    def load_by_npz_name(self, file_name):
        file_path = f"./tmp/mat-{file_name}.npz"
        mat = np.load(file_path)
        self.num_drops = mat['num_drops'][0]
        self.num_classified = mat['num_classified'][0]
        self.random_correct = mat['random_correct']
        self.random_target_confidence = mat['random_target_confidence']
        self.random_loss = mat['random_loss']
        self.high_correct = mat['high_correct']
        self.high_target_confidence = mat['high_target_confidence']
        self.high_loss = mat['high_loss']
        self.low_correct = mat['low_correct']
        self.low_target_confidence = mat['low_target_confidence']
        self.low_loss = mat['low_loss']

    def plot_results(self):
        measures = [
            ('Accuracy', self.random_correct, self.high_correct, self.low_correct),
            # ('Confidence', self.random_target_confidence, self.high_target_confidence, self.low_target_confidence),
            # ('Loss', self.random_loss, self.high_loss, self.low_loss),
        ]

        if self.create_confidence_plot:
            measures.append(('Confidence', self.random_target_confidence, self.high_target_confidence, self.low_target_confidence))

        if self.create_loss_plot:
            measures.append(('Loss', self.random_loss, self.high_loss, self.low_loss))

        colors = ['g', 'r', 'b']
        if self.alternative_colors:
            colors = ['g', 'orange', '#9122D1']

        # create x axis
        x_values = [0]
        for i in range(self.num_drops):
            if self.measure_condition(i + 1):
                x_values.append(i + 1)
            # else ignore as no data for this index has been collected
        #x = np.arange(self.num_drops + 1)  # for 0 drops + 1
        x = np.array(x_values)

        random_acc = self.random_correct / self.num_classified
        low_acc = self.low_correct / self.num_classified
        high_acc = self.high_correct / self.num_classified
        max_diff = np.max(low_acc - random_acc)
        print(max_diff)

        for m in measures:
            plt.figure()
            if self.show_marker is True:
                if self.random_drop:
                    plt.plot(x, m[1] / self.num_classified, color=colors[0], label='rand-drop', ls='-', marker='o', linewidth=1.0)
                plt.plot(x, m[2] / self.num_classified, color=colors[1], label='high-drop', ls='-', marker='+', linewidth=1.0)
                plt.plot(x, m[3] / self.num_classified, color=colors[2], label='low-drop', ls='-', marker='v', linewidth=1.0)
            else:
                if self.random_drop:
                    plt.plot(x, m[1] / self.num_classified, color=colors[0], label='rand-drop', ls='-', linewidth=1.0)
                plt.plot(x, m[2] / self.num_classified, color=colors[1], label='high-drop', ls='-', linewidth=1.0)
                plt.plot(x, m[3] / self.num_classified, color=colors[2], label='low-drop', ls='-', linewidth=1.0)

            if m[0] != 'Loss' and self.plot_guess_line:
                plt.plot(x, np.ones(len(x_values)) * (1/16), color='c', label='random guess', ls='--')
            if self.plot_title:
                plt.title(f"{self.plot_title_prefix_text}: {self.num_classified} pcds, {self.num_drops} drops, {self.steps} step, {'hm updated' if self.update_cam else 'hm not updated'}")
            if m[0] == 'Loss':
                plt.ylim(0.0)
            else:
                ymin = self.diagram_min_y if self.diagram_min_y else 0.0
                ymax = self.diagram_max_y if self.diagram_max_y else 1.05
                plt.ylim(ymin, ymax)
            plt.xlim(0.0)
            if self.plt_max_x:
                plt.xlim(0.0, self.plt_max_x)
            plt.xlabel('Number of Points Dropped', fontsize=18)
            plt.ylabel(m[0], fontsize=18)
            #plt.xticks(np.arange(0, self.num_drops, 100))
            #if m[0] != 'loss':
            #    plt.yticks(np.arange(0, 1., 0.1))
            plt.grid()
            plt.legend()
            suffix = '-hm_updated' if self.update_cam else ''
            os.makedirs("./diagrams", exist_ok=True)
            plt.savefig(f"./diagrams/{self.file_prefix}{self.num_classified}pcds-{self.num_drops}drops-{self.steps}steps-{m[0]}{suffix}.png")
            plt.close()

    def get_area_under_the_curve(self):
        """
        This method calculates the area under the curces for Accuracy, Confidence and Loss for the
        three dropping methods random, low dropping and high dropping.

        The returned 3x3 matrix contains in the rows the measured values for Accuracy, Confidence and Loss in this
        order and in the rows the random, low and high dropping results.
        """
        measures = [
            ('Accuracy', self.random_correct, self.low_correct, self.high_correct),
            ('Confidence', self.random_target_confidence, self.low_target_confidence, self.high_target_confidence),
            ('Loss', self.random_loss, self.low_loss, self.high_loss),
        ]

        # create x axis
        x_values = [0]
        for i in range(self.num_drops):
            if self.measure_condition(i + 1):
                x_values.append(i + 1)
        x_values = np.array(x_values)
        x_num = x_values.shape[0]

        area_matrix = np.zeros((3, 3))

        for i in range(len(measures)):
            for j in range(1, len(measures[i])):
                y_values = measures[i][j][:x_num] / self.num_classified
                total_area = 0.0

                for s in range(0, x_num - 1):
                    # if s == 39:
                    #     print("asd")
                    # calculate are between x[i] and x[i + 1]
                    x = x_values[s]
                    x_next = x_values[s + 1]
                    v = y_values[s]
                    v_next = y_values[s + 1]
                    v_min = min([v, v_next])
                    step_size = (x_next - x)
                    square = step_size * v_min
                    triangle = (abs(v_next - v) * step_size) / 2
                    total_area += square + triangle

                # print("i", i, "j", j)
                area_matrix[i][j - 1] = total_area
                # print("total_are", metric, total_area)

        area_matrix_perc = area_matrix / x_values[-1]  # devide by total possible area (100% accuracy x max_x_value)

        return area_matrix, area_matrix_perc

    def get_loss(self, input, target):
        #return np.square(input.cpu().data.numpy() - target.cpu().data.numpy()).sum()
        return F.cross_entropy(input.view(1, -1), target)


    def get_confidence(self, output, target_idx):
        return torch.nn.functional.softmax(output).cpu()[0, target_idx]


    def update_heatmap(self, input, cam_mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = np.squeeze(heatmap, axis=1)
        heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]  # only one line Change

        self.pcd.colors = o3d.utility.Vector3dVector(
            [heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]] for j in range(input.size(2)))


    def show_pcd(self, draw_points):
        if not self.show_visualization: return
        #print(draw_points)
        self.pcd.points = o3d.utility.Vector3dVector(draw_points[0].transpose(0, 1).cpu().data.numpy())

        if self.create_png:
            self.custom_draw_geometry(self.pcd, point_size=5.0)
        else:
            o3d.visualization.draw_geometries([self.pcd])

    def custom_draw_geometry(self, pcd, x=-190, y=140, point_size=7.0, show_coordinate_frame=False,
                             output_path='./pc-images'):
        # saving does not seem to work on all systems, alternatively save the point cloud including the colors
        # and move them to a system which can create pngs
        create_images = False  # else save the point cloud as ply

        if create_images:
            # Visualize Point Cloud
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)

            opt = vis.get_render_option()  # needs to be called after create_window has been called
            opt.point_size = point_size  # default is 5.0
            opt.show_coordinate_frame = show_coordinate_frame
            opt.background_color = np.asarray([1, 1, 1])

            ctr = vis.get_view_control()
            # ctr.change_field_of_view(step=90)
            ctr.rotate(x, y)
            ctr.scale(-4)

            # Updates
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture image
            time.sleep(1)
            time_suffix = int(round(time.time() * 1000))
            # vis.capture_screen_image(os.path.join(output_path, f"pc-{time_suffix}.png"), do_render=True)

            vis.run()
            # Close
            vis.destroy_window()
        else:
            # alternatively save the point cloud
            o3d.io.write_point_cloud(os.path.join(output_path, f"pc-{self.file_prefix}-{self.iteration_index}-{self.foo}.ply"), pcd)
            self.foo = self.foo + 1
