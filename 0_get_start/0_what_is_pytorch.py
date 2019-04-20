from __future__ import print_function
import torch
#
# # Construct a 5x3 matrix, uninitialized
# x = torch.empty(5, 3)
# print(x)
#
# # Construct a randomly initialized matrix
# x = torch.rand(5, 3)
# print(x)
#
# # Construct a matrix filled zeros and of dtype long
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
#
# # Construct a tensor directly from data
# x = torch.tensor([5.5, 3])
# print(x)
#
# # create a tensor based on an existing tensor.
# # These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x)                                      # result has the same size
#
#
# # Get its size
# # torch.Size is in fact a tuple, so it supports all tuple operations.
# print(x.size())
#
# # Operations
# # Addition: syntax 1
# y = torch.rand(5, 3)
# print(x + y)
# # Addition: syntax 2
# print(torch.add(x, y))
# # Addition: providing an output tensor as argument
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
# # Addition: in-place
# # Any operation that mutates a tensor in-place is post-fixed with an _.
# # For example: x.copy_(y), x.t_(), will change x.
# y.add_(x)
# print(y)
# # support standard NumPy-like indexing
# print(x[:, 1])
#
# # resize/reshape tensor
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())
#
# # If you have a one element tensor, use .item() to get the value as a Python number
# x = torch.randn(1)
# print(x)
# print(x.item())


# ------------NumPy Bridge---------------
# The Torch Tensor and NumPy array will share their underlying memory locations,
# and changing one will change the other.

# # Converting a Torch Tensor to a NumPy Array
# a = torch.ones(5)
# print(a)
#
# b = a.numpy()
# print(b)
# # See how the numpy array changed in value.
# a.add_(1)
# print(a)
# print(b)

# Converting NumPy Array to Torch Tensor
# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)


# -------------CUDA Tensors-------------------
# Tensors can be moved onto any device using the .to method

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    x = torch.randn(1)
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!