import torch
import numpy as np

# Checking if an object is tensor, using torch.is_tensor
x = [1, 2, 3, 4, 5]
x1 = np.array(x)
print(torch.is_tensor(x1))
'''
False
'''
x1_tensor = torch.from_numpy(x1)
print(torch.is_tensor(x1_tensor))
'''
True
'''

# Create linearly-spaced tensor
tensor_linearSpaced = torch.linspace(start=2, end=10, steps=5)
print(tensor_linearSpaced)
'''
tensor([ 2.,  4.,  6.,  8., 10.])
'''

# Random, all-zeros, identity tensors
print(torch.rand(2, 3))
'''
tensor([[0.6098, 0.2553, 0.6182],
        [0.9710, 0.4448, 0.7486]])
'''
print(torch.zeros(2, 3))
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''
print(torch.eye(2, 3))
'''
tensor([[1., 0., 0.],
        [0., 1., 0.]])
'''

# min / max / argmin / argmax
torch.min(x1_tensor)
# tensor(1)
torch.max(x1_tensor)
# tensor(5)
torch.argmin(x1_tensor)
# tensor(0)
torch.argmax(x1_tensor)
# tensor(4)

# Concatenate two tensors: cat()
torch.cat((x1_tensor, x1_tensor))
'''
tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
'''

# Get non-zero elements in tensor.
torch.nonzero(torch.tensor([0, 1, 2, 3, 0]))
'''
tensor([[1],
        [2],
        [3]])
'''

# Split: Split tensor into elements of n
a = torch.split(tensor=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), split_size_or_sections=3)
print(a)
'''
(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8]))
'''
type(a)
'''
<class 'tuple'>
'''

# Transpose: .t() or .transpose()
a = torch.tensor([[1, 2], [3, 4]])
print(a.t())
'''
tensor([[1, 3],
        [2, 4]])
'''
print(a.transpose(dim0=0, dim1=1))
'''
tensor([[1, 3],
        [2, 4]])
'''

# Math functions: add,
a = torch.tensor([[1, 2], [3, 4]])
# Add (hence subtract)
print(torch.add(a, 20))
'''
tensor([[21, 22],
        [23, 24]])
'''
# Multiplication (hence division)
print(torch.mul(a, 3))
'''
tensor([[ 3,  6],
        [ 9, 12]])
'''

# Express linear equation as tensor: Use randn, mul, add
intercept = torch.randn(1)
x = torch.randn(2, 2)
a = 0.7456
# y = (a * x) + intercept
print(torch.add(torch.mul(a, x), intercept))
'''
tensor([[1.9022, 1.1576],
        [2.4524, 2.3719]])
'''

# Ceil / Floor of numbers
torch.ceil(torch.tensor([1.0, 1.1, 1.2, 1.5, 1.8, 1.9, 2.0]))
# tensor([1., 2., 2., 2., 2., 2., 2.])
torch.floor(torch.tensor([1.0, 1.1, 1.2, 1.5, 1.8, 1.9, 2.0]))
# tensor([1., 1., 1., 1., 1., 1., 2.])

# Exponentiation / Logarithm / Power / Sigmoid / Square-Root
torch.exp(torch.tensor([0, 1, 2, 3]))
# tensor([ 1.0000,  2.7183,  7.3891, 20.0855])
torch.log(torch.tensor([0, 1, 2, 3]))
# tensor([  -inf, 0.0000, 0.6931, 1.0986])
torch.pow(input=torch.tensor([0, 1, 2, 3]), exponent=2)
# tensor([0, 1, 4, 9])
torch.sigmoid(torch.tensor([0, 1, 2, 3]))
# tensor([0.5000, 0.7311, 0.8808, 0.9526])
torch.sqrt(torch.tensor([0, 1, 2, 3]))
# tensor([0.0000, 1.0000, 1.4142, 1.7321])
