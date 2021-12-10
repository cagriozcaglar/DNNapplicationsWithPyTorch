### General ###
# Root package
import torch
# Dataset representation and loading
from torch.utils.data import Dataset, DataLoader

### NN API ###
# Computation graph
# Autograd library: https://pytorch.org/docs/stable/autograd.html
import torch.autograd as autograd
# Tensor node in comp graph
from torch import Tensor
# NNs
# NN library: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn
# Layers, activations, and more
# nn.functional library: https://pytorch.org/docs/stable/nn.functional.html
import torch.nn.functional as f
# Optimizers: Adam, Adagrad
# optim library: https://pytorch.org/docs/stable/optim.html
import torch.optim as optim
# hybrid frontend decorator and tracing jit
from torch.jit import script, trace

### Torchscript and JIT ###
# Takes module / function and and example data input, traces the computational steps
# that the data encounters
# torch.jit.trace()
# Decorator used to indicate data-dependent control flow within the code being traced
# Torchscript: https://pytorch.org/docs/stable/jit.html
# @script()


### ONNX ###
# onnx: https://pytorch.org/docs/stable/onnx.html
# Export an ONNX formatted model using a trained model, dummy data, and the desired file name
# torch.onnx.export(model, dummy_data, xxxx.proto)
# Load an ONNX model, check that the model IR is well formed
# model = torch.onnx.load("alexnet.proto")
# torch.onnx.checker.check_model(model)
# Print a human readable representation of the graph
# torch.onnx.helper.printable_graph(model.graph)


### Vision ###
# Vision: https://pytorch.org/vision/stable/index.html
# Vision datasets, architectures, transforms
from torchvision import datasets, models, transforms
# Composable transforms
import torchvision.transforms as transforms

### Distributed Training ###
# Distributed communication
# distributed: https://pytorch.org/docs/stable/distributed.html
import torch.distributed as dist
# Memory sharing processes
# multiprocessing: https://pytorch.org/docs/stable/multiprocessing.html
from torch.multiprocessing import Process

### Tensor Creation ###
size = (2, 3)
# Tensor with independent N(0,1) entries
x = torch.randn(*size)
print(x)
print(type(x))
'''
tensor([[ 0.4117,  0.7128, -0.6331],
        [ 0.3845,  0.6137,  0.4555]])
<class 'torch.Tensor'>
'''
# Tensors with all 1's
x = torch.ones(*size)
print(x)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]])
'''
# Tensors with all 0's
x = torch.zeros(*size)
print(x)
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''
# Create tensor from nested list or ndarray
nested_list = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(nested_list)
print(x)
'''
tensor([[1, 2, 3],
        [4, 5, 6]])
'''
import numpy as np

nd_array = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print(nd_array)
x = torch.tensor(nd_array)
print(x)
'''
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Clone of x
y = x.clone()
print(y)
'''
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Code wrap that stops autograd from tracking tensor history
with torch.no_grad():
    print(y)
'''
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Arg (requires_grad), when set to true, tracks computation history for future derivative calculations
x = torch.randn(*size)
x.requires_grad = True
# Note, with y, which is set to no_grad, requires_grad argument cannot be called.
print(x)
print(type(x))
'''
tensor([[-0.2790,  0.4273,  1.3725],
        [ 0.5989,  0.4031, -0.2336]], requires_grad=True)
<class 'torch.Tensor'>
'''

### Tensor Dimensionality ###
nd_array = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
x = torch.tensor(nd_array)
print(x)
'''
tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Return tuple-like objects of dimensions
print(f"x.size(): {x.size()}")
print(type(x.size()))
'''
x.size(): torch.Size([2, 3])
<class 'torch.Size'>
'''
# Concatenates tensors along dim
z = torch.cat([x, x], dim=0)
print(z)
print(type(z))
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
<class 'torch.Tensor'>
'''
# Reshapes z into size (a,b,...)
t = z.view(2, 6)
print(t)
print(type(t))
'''
tensor([[1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]], dtype=torch.int32)
<class 'torch.Tensor'>
'''
# Reshapes z into size (a,b) for some a
# Useful when you are sure of all dimensions except one
t = z.view(-1, 6)
print(t)
print(type(t))
'''
tensor([[1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]], dtype=torch.int32)
<class 'torch.Tensor'>
'''
# Swaps dimensions a and b (in the example below, change dimensions 0 and 1)
t = x.transpose(0, 1)
print(t)
'''
tensor([[1, 4],
        [2, 5],
        [3, 6]], dtype=torch.int32)
'''
# Permutes dimensions
dims = [0, 1]
t = z.permute(*dims)
print(t)
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Tensors with added axis
t = z.unsqueeze(1)
print(t)
'''
tensor([[[1, 2, 3]],
        [[4, 5, 6]],
        [[1, 2, 3]],
        [[4, 5, 6]]], dtype=torch.int32)
'''
# (a,b,c) tensor -> (a,b,1,c) tensor
t = z.unsqueeze(2)
print(t)
'''
tensor([[[1],
         [2],
         [3]],
        [[4],
         [5],
         [6]],
        [[1],
         [2],
         [3]],
        [[4],
         [5],
         [6]]], dtype=torch.int32)
'''
# Removes all dimensions of size 1: (a,1,b,1) -> (a,b)
r = t.squeeze()
print(r)
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''
# Removes specified dimension of size 1: (a,b,1) -> (a,b) (dim=2 (zero-indexed) is removed)
r = t.squeeze(dim=2)
print(r)
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
'''

### Tensor Algebra ###
# A matrix (matrix_A) of dim 2x3
A = [[1., 2., 3.], [4., 5., 6.]]
matrix_A = torch.tensor(A)
print(matrix_A)
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]])
'''
# A matrix (matrix_B) of dim 3x2
B = [[7., 8.], [9., 10.], [11., 12.]]
matrix_B = torch.tensor(B)
print(matrix_B)
'''
tensor([[ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
'''
# Matrix multiplication ( matrix_A (2x3)  * matrix_B (3x2) = matrix_mult (2x2) )
matrix_mult = matrix_A.mm(matrix_B)
print(matrix_mult)
'''
tensor([[ 58.,  64.],
        [139., 154.]])
'''
# Matrix-Vector multiplication
# A vector (vector_C) of dim 1x3
C = [1., 2., 3]
vector_C = torch.tensor(C)
print(vector_C)
'''
tensor([1., 2., 3.])
'''
# Multiply matrix_A (of size 2x3) with vector_C (of size 1x3)
matrix_vector_mult = matrix_A.mv(vector_C)
print(matrix_vector_mult)
'''
tensor([14., 32.])
'''
# Transpose (of a vector or a matrix)
# Matrix transpose: matrix_A (2x3) => matrix_A_transpose (3x2)
matrix_A_transpose = matrix_A.t()
print(matrix_A_transpose)
'''
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
'''
# Vector transpose: vector_C (1x3) => vector_C_transpose (1x3)
# Same vector is returned. This is because the vector tensor has a dimension of 1. You can only take the transpose of a
# tensor with dimension >2.
vector_C_transpose = vector_C.t()
print(vector_C_transpose)
'''
tensor([1., 2., 3.])
'''

### GPU Usage ###
# Check for cuda
torch.cuda.is_available()
# Move x's data from CPU to GPU and return new object
# x_newObjectForCuda = x.cuda()
# Move x's data from GPU to CPU and return new object
# x = x_newObjectForCuda.cpu()
# Device-agnostic code and modularity
import argparse
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
print(args)
'''
Namespace(disable_cuda=False, device=None)
'''
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
# Recursively convert their parameters and buffers to device specific tensors
from collections import OrderedDict

net = nn.Sequential(OrderedDict([('fc1', nn.Linear(3, 1))]))
device = torch.device('cpu')
net.to(device)
# Copy your tensors to a device
x_newDevice = x.to(device)

### Deep Learning ###
# Fully connected layer from in_features to out_features units
input_feature_count = 200
output_feature_count = 50
fully_connected_layer = nn.Linear(
    in_features=input_feature_count,
    out_features=output_feature_count
)
print(fully_connected_layer)
'''
Linear(in_features=200, out_features=50, bias=True)
'''
# X-dimensional convolutional layer from in_channels to out_channels,
# where X is one of {1,2,3} (nn.ConvXd, where X is in {1,2,3})
# and kernel size is kernel_size
input_channel_count = 200
output_channel_count = 50
kernel_size = (3, 5)
conv_1d_layer = nn.Conv1d(
    in_channels=input_channel_count,
    out_channels=output_channel_count,
    kernel_size=kernel_size
)
print(conv_1d_layer)
'''
Conv1d(200, 50, kernel_size=(3, 5), stride=(1,))
'''
conv_2d_layer = nn.Conv2d(
    in_channels=input_channel_count,
    out_channels=output_channel_count,
    kernel_size=kernel_size
)
print(conv_2d_layer)
'''
Conv2d(200, 50, kernel_size=(3, 5), stride=(1, 1))
'''
conv_3d_layer = nn.Conv3d(
    in_channels=input_channel_count,
    out_channels=output_channel_count,
    kernel_size=kernel_size
)
print(conv_3d_layer)
'''
Conv3d(200, 50, kernel_size=(3, 5), stride=(1, 1, 1))
'''
# X-dimension pooling layer (MaxPoolXd, where X is in {1,2,3})
max_pool_1d = nn.MaxPool1d(kernel_size=(1,))
print(max_pool_1d)
'''
MaxPool1d(kernel_size=(1,), stride=(1,), padding=0, dilation=1, ceil_mode=False)
'''
max_pool_2d = nn.MaxPool2d(kernel_size=(1,))
print(max_pool_2d)
'''
MaxPool2d(kernel_size=(1,), stride=(1,), padding=0, dilation=1, ceil_mode=False)
'''
max_pool_3d = nn.MaxPool3d(kernel_size=(1,))
print(max_pool_3d)
'''
MaxPool3d(kernel_size=(1,), stride=(1,), padding=0, dilation=1, ceil_mode=False)
'''
# Batch norm layer, using BatchNormXd
batch_norm_1d = nn.BatchNorm1d(2)
print(batch_norm_1d)
'''
BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
'''
batch_norm_2d = nn.BatchNorm2d(2)
print(batch_norm_2d)
'''
BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
'''
batch_norm_3d = nn.BatchNorm3d(2)
print(batch_norm_3d)
'''
BatchNorm3d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
'''
# Recurrent layers: RNN, LSTM, GRU
rnn_layer = nn.RNN(
    input_size=200,
    hidden_size=50
)
print(rnn_layer)
'''
RNN(200, 50)
'''
lstm_layer = nn.LSTM(
    input_size=200,
    hidden_size=50
)
print(lstm_layer)
'''
LSTM(200, 50)
'''
gru_layer = nn.GRU(
    input_size=200,
    hidden_size=50
)
print(gru_layer)
'''
GRU(200, 50)
'''
# Dropout layer for any dimensional input
dropout_layer_1d = nn.Dropout(
    p=0.5,
    inplace=False
)
print(dropout_layer_1d)
'''
Dropout(p=0.5, inplace=False)
'''
# 2-dimensional channel-wise drop-out
dropout_layer_2d = nn.Dropout2d(
    p=0.5,
    inplace=False
)
print(dropout_layer_2d)
'''
Dropout2d(p=0.5, inplace=False)
'''
# 3-dimensional channel-wise drop-out
dropout_layer_3d = nn.Dropout3d(
    p=0.5,
    inplace=False
)
print(dropout_layer_3d)
'''
Dropout3d(p=0.5, inplace=False)
'''
# Tensor-wise mapping from indices to embedding vectors
mapping_from_indices_to_embedding_vectors = nn.Embedding(
    num_embeddings=200,
    embedding_dim=50
)
print(mapping_from_indices_to_embedding_vectors)
'''
Embedding(200, 50)
'''

### Loss Functions ###
# nn.X, where X is the loss function name
l1_loss = nn.L1Loss()
print(l1_loss)
'''
L1Loss()
'''
nn.MSELoss()
nn.CrossEntropyLoss()
nn.CTCLoss()
nn.NLLLoss()
nn.PoissonNLLLoss()
nn.KLDivLoss()
nn.BCELoss()
nn.BCEWithLogitsLoss()
nn.MarginRankingLoss()
nn.HingeEmbeddingLoss()
nn.MultiLabelMarginLoss()
nn.SmoothL1Loss()
nn.SoftMarginLoss()
nn.MultiLabelSoftMarginLoss()
nn.CosineEmbeddingLoss()
nn.MultiMarginLoss()
nn.TripletMarginLoss()

### Activation Functions ###
# nn.X, where X is the activation function name
relu_activation = nn.ReLU()
print(relu_activation)
'''
ReLU()
'''
nn.ReLU6()
nn.ELU()
nn.SELU()
nn.PReLU()
nn.LeakyReLU()
nn.RReLU()
nn.CELU()
nn.GELU()
nn.Threshold(threshold=0.1, value=4)
nn.Hardshrink()
nn.Hardtanh()
nn.Sigmoid()
nn.LogSigmoid()
nn.Softplus()
nn.Softshrink()
nn.Softsign()
nn.Tanh()
nn.Tanhshrink()
nn.Softmin()
nn.Softmax()
nn.Softmax2d()
nn.LogSoftmax()
nn.AdaptiveLogSoftmaxWithLoss(in_features=200, n_classes=5, cutoffs=[1, 2])


### Optimizers ###
# Create optimizer: optim.X, where X is the optimizer name
model_parameters = {}
# Create Optimizer
optimizer_sgd = optim.SGD(params=model_parameters, lr=0.01, momentum=0.9)
# Update weights
optimizer_sgd.step()
# Other optimizers: optim.X, where X is the optimizer name
optimizer_adam = optim.Adam(params=[{1: 2}], lr=0.0001)
optim.Adadelta(params=[{}])
optim.Adagrad(params=[{}])
optim.AdamW(params={})
optim.SparseAdam(params={})
optim.Adamax(params={})
optim.ASGD(params={})
optim.LBFGS(params={})
optim.RMSprop(params={})
optim.Rprop(params={})


### Learning Rate Scheduling ###
scheduler = optim.lr_scheduler
# scheduler.step()
# optim.lr_scheduler.X, where X is schedule name
# Several schedule examples
optim.lr_scheduler.LambdaLR(optimizer=optimizer_sgd, lr_lambda=lambda integer: integer+0.1)


### Data Utilities ###

### Datasets ###
# Abstract class representing dataset
# Dataset
# Labelled dataset in the form of tensors
# TensorDataset
# Concatenation of Datasets
# Concat Dataset


### DataLoaders and DataSamplers ###
# Loads data batches agnosic of structure of individual data points
# DataLoader(dataset, batch+size=1, ...)
# Abstract class dealing with ways to sample from dataset
# sampler.Sampler(dataset, ...)
# X is in {Sequential, Random, SubsetRandom, WeightedRandom Batch, Distributed}
# sampler.XSampler where ...

# x = torch.rand(2, 4)
# print(x)