# -------------------neural networks--------------------
# nn depends on autograd to define models and differentiate them
# An nn.Module contains layers, and a method forward(input)that returns the output.

# ######################################################################
# A typical training procedure for a neural network is as follows:
# ################################################################
# 1. Define the neural network that has some learnable parameters (or weights)
# 2. Iterate over a dataset of inputs
# 3. Process input through the network
# 4. Compute the loss (how far is the output from being correct)
# 5. Propagate gradients back into the network’s parameters
# 6. Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

# Define the network
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel(gray image), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# define the forward function, and the backward function (where gradients are computed)
# is automatically defined for you using autograd.
# You can use any of the Tensor operations in the forward function

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# Let try a random 32x32 input(size of LeNet).
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# ---------------torch.nn only supports mini-batches-------
# nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.


# --------------------Loss Function--------------------
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print('loss: ', loss)

# graph of computations:
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
# -> view -> linear -> relu -> linear -> relu -> linear
# -> MSELoss
# -> loss

# backward
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# ----------------------Backprop-----------------------
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# -----------------------Update the weights--------------------
# Stochastic Gradient Descent(SGD)
# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# package: torch.optim implements different update rules
# such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update