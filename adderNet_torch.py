import torch
import torch.nn as nn

class AdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # x:      (N, D)
        # weight: (D, M)

        diff = x[:, :, None] - weight[None, :, :]   # (N, D, M)

        ctx.save_for_backward(diff)

        out = -torch.sum(torch.abs(diff), dim=1)    # (N, M)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors

        # grad_output: (N, M)
        g = grad_output[:, None, :]                 # (N, 1, M)

        # surrogate derivative for abs(diff)
        clipped = torch.clamp(diff, -1, 1)          # (N, D, M)

        # because output = -sum(abs(diff))
        grad_x = torch.sum(-clipped * g, dim=2)     # (N, D)

        # because diff = x - weight, derivative wrt weight flips sign
        grad_weight = torch.sum(clipped * g, dim=0) # (D, M)

        return grad_x, grad_weight


class AdderLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return AdderFunction.apply(x, self.weight)
    

class AdderNet(nn.Module):
    def __init__(self, layer_sizes, use_batchnorm=True):
        super().__init__()

        layers = []

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]

            layers.append(AdderLayer(in_features, out_features))

            # Add BatchNorm after hidden layers only, not after final output layer
            if use_batchnorm and i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_features))

        self.net = nn.Sequential(*layers)

        # Optional final affine scale/bias
        self.out_scale = nn.Parameter(torch.ones(1, layer_sizes[-1]))
        self.out_bias = nn.Parameter(torch.zeros(1, layer_sizes[-1]))

    def forward(self, x):
        x = self.net(x)
        x = self.out_scale * x + self.out_bias
        return x