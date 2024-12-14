import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        if inputs.shape[0] == targets.shape[0]:
            for x, y in zip(inputs, targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()
        else:
            bs_size = targets.shape[0]
            for x, y in zip(inputs[:bs_size], targets):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))




class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, return_out=False):
        assert inputs.shape[0] == targets.shape[0]
        
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = cm(inputs, targets, self.features, self.momentum)
        outputs /= self.temp

        if return_out:
            return outputs
        loss = F.cross_entropy(outputs, targets)

        return loss
            

            
