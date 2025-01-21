import torch


class MinMaxScaler(object):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, data):
        return (data - self.min) / (self.max - self.min)


class MinMaxNegScaler(object):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, data):
        data = (data - self.min) / (self.max - self.min)
        return data * 2 - 1


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std


class RandomFlip(object):
    def __init__(self, dims=(0,), p=0.5):
        self.dims = dims
        self.p = p

    def __call__(self, data):
        rand = torch.rand(1).item()
        return torch.flip(data, dims=self.dims) if (rand <= self.p) else data


class RandomRot90(object):
    def __init__(self, dims=(0, 1)):
        self.dims = dims

    def __call__(self, data):
        rand = torch.randint(0, 4, (1,)).item()
        return torch.rot90(data, k=rand, dims=self.dims)


class ReshapeTransform(object):
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, data):
        return torch.reshape(data, self.new_shape)


class ReduceDimension(object):
    def __init__(self, op, dim=0):
        self.op = op
        self.dim = dim

    def __call__(self, data):
        return self.op(data, dim=self.dim)


class LogTransform(object):
    def __call__(self, data):
        return torch.log(data)
