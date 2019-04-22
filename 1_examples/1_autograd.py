# -*- coding: utf-8 -*-
# Autograd: define a computational graph;
# nodes in the graph will be Tensors,
# and edges will be functions that produce output Tensors from input Tensors.

import torch

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()


# -------------------Defining new autograd functions--------------------
import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()


# ---------------------Static Graphs----------------------
# import tensorflow as tf
# import numpy as np
#
# # First we set up the computational graph:
#
# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # Create placeholders for the input and target data; these will be filled
# # with real data when we execute the graph.
# x = tf.placeholder(tf.float32, shape=(None, D_in))
# y = tf.placeholder(tf.float32, shape=(None, D_out))
#
# # Create Variables for the weights and initialize them with random data.
# # A TensorFlow Variable persists its value across executions of the graph.
# w1 = tf.Variable(tf.random_normal((D_in, H)))
# w2 = tf.Variable(tf.random_normal((H, D_out)))
#
# # Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# # Note that this code does not actually perform any numeric operations; it
# # merely sets up the computational graph that we will later execute.
# h = tf.matmul(x, w1)
# h_relu = tf.maximum(h, tf.zeros(1))
# y_pred = tf.matmul(h_relu, w2)
#
# # Compute loss using operations on TensorFlow Tensors
# loss = tf.reduce_sum((y - y_pred) ** 2.0)
#
# # Compute gradient of the loss with respect to w1 and w2.
# grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
#
# # Update the weights using gradient descent. To actually update the weights
# # we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# # in TensorFlow the the act of updating the value of the weights is part of
# # the computational graph; in PyTorch this happens outside the computational
# # graph.
# learning_rate = 1e-6
# new_w1 = w1.assign(w1 - learning_rate * grad_w1)
# new_w2 = w2.assign(w2 - learning_rate * grad_w2)
#
# # Now we have built our computational graph, so we enter a TensorFlow session to
# # actually execute the graph.
# with tf.Session() as sess:
#     # Run the graph once to initialize the Variables w1 and w2.
#     sess.run(tf.global_variables_initializer())
#
#     # Create numpy arrays holding the actual data for the inputs x and targets
#     # y
#     x_value = np.random.randn(N, D_in)
#     y_value = np.random.randn(N, D_out)
#     for _ in range(500):
#         # Execute the graph many times. Each time it executes we want to bind
#         # x_value to x and y_value to y, specified with the feed_dict argument.
#         # Each time we execute the graph we want to compute the values for loss,
#         # new_w1, and new_w2; the values of these Tensors are returned as numpy
#         # arrays.
#         loss_value, _, _ = sess.run([loss, new_w1, new_w2],
#                                     feed_dict={x: x_value, y: y_value})
#         print(loss_value)