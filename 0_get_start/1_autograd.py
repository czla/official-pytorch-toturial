import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor operation
y = x + 2
print(y)
# y was created as a result of an operation, so it has a grad_fn.
print(y.grad_fn)
# Do more operations on y
z = y * y * 3
out = z.mean()

print(z, out)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
# The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


# ----------------Gradients--------------
# backprop：Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.)).
# Print gradients d(out)/dx
out.backward()
print(x.grad)

# Stop autograd from tracking history on Tensors with .requires_grad=True
# by wrapping the code block in with torch.no_grad()
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)