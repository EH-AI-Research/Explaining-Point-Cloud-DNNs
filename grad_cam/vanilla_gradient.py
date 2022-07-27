import torch
import numpy as np

class Gradient:
    def __init__(self, model, type, cuda=True, normalize=True):
        self.type = type
        self.model = model
        self.model.eval()
        self.classifier_output = None
        self.cuda = cuda
        self.normalize = normalize

    def __call__(self, input, target):
        #if self.cuda:
        #    input = input.cuda()
        #else:
        #    input = input
        input.requires_grad = True
        self.model.zero_grad()
        output = self.model(input)
        self.classifier_output = output
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, target] = 1
        output.backward(gradient=grad_outputs)
        input.requires_grad = False

        # foobar = input.grad.clone()
        #mask = (input.grad.clone() * input)
        mask = input.grad.clone()
        mask = torch.max(mask, dim=1)[0]

        if self.type == 'positive':
            mask = mask.clamp(min=0)
        elif self.type == 'negative':
            mask = mask.clamp(max=0)
        elif self.type == 'absolute':
            mask = mask.abs()
        elif self.type == 'plain':
            pass
        else:
            raise Exception("Unknown type")

        mask = mask.cpu().data.numpy()
        if self.normalize:
            mask = mask - np.expand_dims(np.min(mask, axis=1), 0).T
            mask = mask / np.expand_dims(np.max(mask, axis=1), 0).T
        return mask[0]
