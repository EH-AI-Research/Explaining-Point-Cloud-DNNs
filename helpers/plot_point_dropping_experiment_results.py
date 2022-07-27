import open3d as o3d
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
import os

class PlotPointDropExperimentResults:

    def __init__(self, steps):
        self.steps = steps

    def get_file_name(self, file_name):
        print("os.path.ab", os.path.abspath(f"./tmp/{file_name}.npz"))
        return f"./tmp/{file_name}.npz"

    def load_results(self, file_name):
        mat = np.load(self.get_file_name(file_name))
        return {
            'num_drops': mat['num_drops'][0],
            'num_classified': mat['num_classified'][0],
            'random_correct': mat['random_correct'],
            'random_target_confidence': mat['random_target_confidence'],
            'random_loss': mat['random_loss'],
            'high_correct': mat['high_correct'],
            'high_target_confidence': mat['high_target_confidence'],
            'high_loss': mat['high_loss'],
            'low_correct': mat['low_correct'],
            'low_target_confidence': mat['low_target_confidence'],
            'low_loss': mat['low_loss'],
        }

    def measure_condition(self, i, num_drops):
        return i % self.steps == 0 or i == num_drops - 1

    def plot_results(self, files, metrics, labels, colors, markers, linestyle, title="Default title", save_to="plot",
                     plt_max_x=None, plt_min_y=None, plt_max_y=None, max_values=None, random_guess=True):
        assert len(files) == len(metrics) == len(colors) == len(labels) == len(markers)

        plt.figure(figsize=(8, 4))
        x = None

        for fid in range(len(files)):
            exp_result = self.load_results(files[fid])
            num_drops = exp_result['num_drops']
            num_classified = exp_result['num_classified']

            if x is None:
                # create x axis
                x_values = [0]
                for i in range(num_drops):
                    if self.measure_condition(i + 1, num_drops):
                        x_values.append(i + 1)
                x = np.array(x_values)

            for cid in range(len(metrics[fid])):
                metric_name = metrics[fid][cid]
                acc = exp_result[metric_name] / num_classified
                if max_values:
                    x = x[0:max_values]
                    acc = acc[0:max_values]
                plt.plot(x, acc, color=colors[fid][cid], label=labels[fid][cid], ls=linestyle[fid][cid], marker=markers[fid][cid], linewidth=1.0)

        if random_guess:
            random_guess_y = np.ones(len(x_values))
            if max_values:
                random_guess_y = random_guess_y[0:max_values]
            plt.plot(x, random_guess_y * (1 / 16), color='c', label='random guess', ls='--')
            plt.title(title)

        ymin = plt_min_y if plt_min_y else 0.0
        ymax = plt_max_y if plt_max_y else 1.05
        plt.ylim(ymin, ymax)

        plt.xlim(0.0)
        if plt_max_x:
            plt.xlim(0.0, plt_max_x)
        plt.xlabel('Number of Points Dropped', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)

        plt.grid()
        plt.legend()
        plt.tight_layout()
        os.makedirs("./diagrams", exist_ok=True)
        plt.savefig(f"./diagrams/{save_to}.png", transparent=True, dpi=300)
        plt.close()