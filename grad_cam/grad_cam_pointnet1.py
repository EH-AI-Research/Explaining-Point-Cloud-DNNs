import numpy as np
import torch
from torch.autograd import Variable


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():

            x = module(x)

            if name in self.target_layers:
                x.register_hook(self.save_gradient)

                outputs += [x]
            #if name == '7':
              #  x = torch.max(x, 2, keepdim=True)[0]
              #  x = x.view(-1, 1024)

        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features.feats, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        # F.log_softmax(x, dim=1)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, counterfactual=False, use_cuda=True, normalize=True, disable_relu=False):
        self.model = model
        self.model.eval()
        if use_cuda:
            self.model.cuda()
        self.cuda = use_cuda
        self.extractor = ModelOutputs(self.model, target_layer_names)
        self.counterfactual = counterfactual
        self.normalize = normalize
        self.disable_relu = disable_relu

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        self.classifier_output = output

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.feats.zero_grad()
        self.model.classifier.zero_grad()
        # one_hot.backward(retain_variables=True)
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        if not self.disable_relu:
            cam = np.maximum(cam, 0)  # ReLU

        if self.normalize:
            #cam = cv2.resize(cam, (224, 224))
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                # in the drop point experiment the gradients are at some time 0 leading to a division by zero error
                cam = cam / np.max(cam)

        return cam