"""
Convolutions for Images
"""
import torch
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_6_t
import torchvision
from torchvision import datasets, transforms


def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Compute 2-D cross-correlation (convolution) of input tensor X and kernel tensor K
    :param X: Input tensor X
    :param K: Kernel tensor K
    :return: Output tensor Y
    """
    x_h, x_w = X.shape
    k_h, k_w = K.shape
    Y: torch.Tensor = torch.zeros((x_h - k_h + 1), (x_w - k_w + 1))
    for i in range(Y.shape[0]):  # x-axis: height
        for j in range(Y.shape[1]):  # y-axis: width
            Y[i, j] = (X[i:i + k_h, j:j + k_w] * K).sum()
    return Y


# Example convolution using corr2d() method
X = torch.tensor([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
K = torch.tensor([
    [0, 1],
    [2, 3]
])
print(corr2d(X, K))
# tensor([[19., 25.],
#         [37., 43.]])


class Conv2D(Module):
    """
    2-dimensional convolutional layer based on the corr2d() function defined above
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = Parameter(torch.rand(kernel_size))
        self.bias = Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# Use Conv2D
print(Conv2D((2, 3)))
# Conv2D()


# Object Edge Detection in Images
# Input tensor X: Size 6x8, middle 4 columns are black (0), and the rest are white (1)
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])
# Kernel tensor: Size 1x2, [1, -1]. If horizontally adjacent elements are same, the output is 0. Otherwise, non-zero.
K = torch.tensor([[1.0, -1.0]])
print(K)
# tensor([[ 1., -1.]])
# After convolution, output tensor Y.
Y = corr2d(X, K)
print(Y)
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])


# Learning a Kernel
# Same example above, but we will learn the kernel using gradient descent
# Construct a 2-D convolutonal layer with 1 output channel and a kernel of shape (1, 2).
# For simplicity, ignore bias.
# Use built-in Conv2D class.
conv2d = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(1, 2),
    bias=False
)
# The 2-D Conv layer uses 4-D input and output in the format of (example, channel, height, width), where
# the batch size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
# Learning rate of GD
learning_rate = 3e-2
# GD, only 10 iterations, instead of using a convergence criteria
for i in range(10):
    Y_hat = conv2d(X)
    loss = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    loss.sum().backward()
    # Update kernel: GD equation: w = w - lr * gradient
    conv2d.weight.data[:] -= learning_rate * conv2d.weight.grad
    print(f'Epoch {i+1}, loss {loss.sum(): .3f}')
# Epoch 1, loss  9.225
# Epoch 2, loss  4.673
# Epoch 3, loss  2.486
# Epoch 4, loss  1.385
# Epoch 5, loss  0.802
# Epoch 6, loss  0.478
# Epoch 7, loss  0.292
# Epoch 8, loss  0.181
# Epoch 9, loss  0.113
# Epoch 10, loss  0.072
# Notice that the error has dropped to a small value after 10 iterations
# When we check the learned kernel value, we see that it is very close to the kernel tensor K ([1,-1]) defined earlier
print(conv2d.weight.data.reshape((1, 2)))
# tensor([[ 0.9700, -1.0020]])


"""
Padding and Stride
"""


# Padding
def comp_conv2d(conv2d: torch.nn.Conv2d, X: torch.Tensor) -> torch.Tensor:
    """
    We define a convenience function to calculate the convolutional layer.
    It initialized the conv layer weights and performs corresponding dimensionality elevations and reductions on the
    input and output
    :param conv2d: convolutional layer of type torch.nn.Conv2d
    :param X: input tensor
    :return: output tensor after multiplication with kernel with padding
    """
    # (1, 1) indicates: batch size = 1, number of channels = 1
    X = X.reshape((1, 1) + X.shape)
    print(X.shape)
    print(X)
    # Apply convolutional layer
    Y = conv2d(X)
    print(Y.shape)
    print(Y)
    # Exclude first two dimensions that do not interest us: batch size and channels
    Y_reshaped = Y.reshape(Y.shape[2:])
    print(Y_reshaped.shape)
    print(Y)
    return Y_reshaped


# Padding using comp_conv2d method above
# Here, 1 row and 1 column is padded on either side, so, a total of 2 rows or columns are added
# The input size and output size are the same, because:
# X: x_h x x_w (8 x 8)
# K: k_h x k_w (3 x 3)
# Y: (x_h - k_h + p_h + 1) x (x_w - k_w + p_w + 1) = (8 - 3 + 2 + 1) x (8 - 3 + 2 + 1) = 8 x 8
conv2d = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=1
)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)
# torch.Size([1, 1, 8, 8])
# tensor([[[[0.9942, 0.5337, 0.0283, 0.6322, 0.5175, 0.7953, 0.9701, 0.1651],
#           [0.8345, 0.1061, 0.9995, 0.3899, 0.0482, 0.6334, 0.6290, 0.9108],
#           [0.7514, 0.1679, 0.7212, 0.6287, 0.4981, 0.2598, 0.1984, 0.4871],
#           [0.7965, 0.5316, 0.6568, 0.2492, 0.6471, 0.3526, 0.7271, 0.9207],
#           [0.0400, 0.4701, 0.1082, 0.0693, 0.4953, 0.2436, 0.3087, 0.8002],
#           [0.4578, 0.8807, 0.4472, 0.1378, 0.7126, 0.1468, 0.8770, 0.4157],
#           [0.3262, 0.6732, 0.0296, 0.4918, 0.8715, 0.4665, 0.9219, 0.6948],
#           [0.6412, 0.5966, 0.3571, 0.1770, 0.2068, 0.3806, 0.4620, 0.4735]]]])
# torch.Size([1, 1, 8, 8])
# tensor([[[[0.3518, 0.6393, 0.3850, 0.4774, 0.4248, 0.4068, 0.6353, 0.4165],
#           [0.7308, 0.5371, 0.4945, 0.7615, 0.5634, 0.6546, 0.5948, 0.2866],
#           [0.7333, 0.5359, 0.7719, 0.5251, 0.4302, 0.7922, 0.7490, 0.6919],
#           [0.6149, 0.2871, 0.6090, 0.4882, 0.4616, 0.4185, 0.5323, 0.5154],
#           [0.7587, 0.6499, 0.6354, 0.5355, 0.5743, 0.6664, 0.6484, 0.6801],
#           [0.4459, 0.5136, 0.4623, 0.4903, 0.7057, 0.6312, 0.7046, 0.6814],
#           [0.6200, 0.7752, 0.4296, 0.3880, 0.5965, 0.3780, 0.7304, 0.4027],
#           [0.3932, 0.4080, 0.2062, 0.4509, 0.4845, 0.3454, 0.5124, 0.3739]]]],
#        grad_fn=<ThnnConv2DBackward0>)
# torch.Size([8, 8])
# tensor([[[[0.3518, 0.6393, 0.3850, 0.4774, 0.4248, 0.4068, 0.6353, 0.4165],
#           [0.7308, 0.5371, 0.4945, 0.7615, 0.5634, 0.6546, 0.5948, 0.2866],
#           [0.7333, 0.5359, 0.7719, 0.5251, 0.4302, 0.7922, 0.7490, 0.6919],
#           [0.6149, 0.2871, 0.6090, 0.4882, 0.4616, 0.4185, 0.5323, 0.5154],
#           [0.7587, 0.6499, 0.6354, 0.5355, 0.5743, 0.6664, 0.6484, 0.6801],
#           [0.4459, 0.5136, 0.4623, 0.4903, 0.7057, 0.6312, 0.7046, 0.6814],
#           [0.6200, 0.7752, 0.4296, 0.3880, 0.5965, 0.3780, 0.7304, 0.4027],
#           [0.3932, 0.4080, 0.2062, 0.4509, 0.4845, 0.3454, 0.5124, 0.3739]]]],
#        grad_fn=<ThnnConv2DBackward0>)
# torch.Size([8, 8])

# When height & width of kernel are different, we can use different padding numbers for height and width,
# in order to make input and output have the same height and weight
conv2d = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(5, 3),
    padding=(2, 1)
)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])
# How?
# Here, 2 row and 1 column is padded on either side, so, a total of 4 rows or 2 columns are added
# The input size and output size are the same, because:
# X: x_h x x_w (8 x 8)
# K: k_h x k_w (5 x 3)
# Y: (x_h - k_h + p_h + 1) x (x_w - k_w + p_w + 1) = (8 - 5 + 4 + 1) x (8 - 3 + 2 + 1) = 8 x 8


# Stride
# If stride height and width is (s_h, s_w) and padding height and width is (p_h, p_w), the output shape is:
# floor( (x_h - k_h + p_h + s_h) / s_h)  x  floor( (x_w - k_w + p_w + s_w) / s_w)

# Here, we set stride height / width to 2, thus halving the input height / width
conv2d = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=1,
    stride=2
)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([4, 4])
# How?
# X: x_h x x_w (8 x 8)
# K: k_h x k_w (3 x 3)
# Y: floor( (x_h - k_h + p_h + s_h) / s_h)  x  floor( (x_w - k_w + p_w + s_w) / s_w)
#  = floor( (8-3+1+2)/2 )  x  floor( (8-3+1+2)/2 )
#  = floor( 8/2 )  x  floor( 8/2 )
#  = 4 x 4

# Example with uneven strides, padding, kernel
conv2d = torch.nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(3, 5),
    padding=(0, 1),
    stride=(3, 4)
)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([2, 2])
# How?
# X: x_h x x_w (8 x 8)
# K: k_h x k_w (3 x 5)
# Y: floor( (x_h - k_h + p_h + s_h) / s_h)  x  floor( (x_w - k_w + p_w + s_w) / s_w)
#  = floor( (8-3+0+3)/3 )  x  floor( (8-5+1+4)/4 )
#  = floor( 8/3 )  x  floor( 8/4 )
#  = 2 x 2


"""
Multiple Input and Multiple Output Channels
"""


# Multiple Input Channels
def corr2d_multi_input_channels(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-correlation (convolution) with multiple input channels
    :param X: Input tensor of size x_channels * x_h * x_w
    :param K: Kernel tensor of size x_channels * k_h * k_w
    :return: Y: Output tensor
    """
    # Iterate through channels (0-th dimension) of input tensor X and kernel tensor K.
    # Then, add them together
    # Note: We are using corr2d method defined in this file, used to compute 2-D cross correlation
    return sum(corr2d(x, k) for x, k in zip(X, K))


# Example: Multiple input channels
X = torch.tensor([
    # Channel 1
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ],
    # Channel 2
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
])
print(X.shape)
# torch.Size([2, 3, 3])
K = torch.tensor([
    # Channel 1
    [
        [0, 1],
        [2, 3]
    ],
    # Channel 2
    [
        [1, 2],
        [3, 4]
    ]
])
print(K.shape)
# torch.Size([2, 2, 2])
Y = corr2d_multi_input_channels(X, K)
print(Y)
# tensor([[ 56.,  72.],
#         [104., 120.]])
print(Y.shape)
# torch.Size([2, 2])


# Multiple output channels
def corr2d_multi_input_output_channels(X, K):
    """
    Compute cross-correlation (convolution) with multiple input channels and output channels
    :param X: Input tensor of size x_channels (c_i) * x_h * x_w
    :param K: Kernel tensor of size x_output_channels (c_o) * x_channels (c_i) * k_h * k_w
    :return: Y: Output tensor
    """
    # Iterate through channels (0-th dimension) of kernel tensor K, and each time, perform cross-correlation
    # operations with input tensor X.
    # Then, stack results together.
    # Note: We are using corr2d method defined in this file, used to compute 2-D cross correlation
    return torch.stack([corr2d_multi_input_channels(X, k) for k in K], 0)


# Example
# Initial 3-D kernel tensor K
print(K)
# tensor([[[0, 1],
#          [2, 3]],
#         [[1, 2],
#          [3, 4]]])
print(K.shape)
# torch.Size([2, 2, 2])
# Add 4-th dimension, output channel, to kernel tensor K, in the 0th dimension
K = torch.stack((K, K+1, K+2), 0)
print(K)
# tensor([
#     # Output channel 1
#     [
#         # Input channel 1
#         [
#             [0, 1],
#             [2, 3]
#         ],
#         # Input channel 2
#         [
#             [1, 2],
#             [3, 4]
#         ]
#     ],
#     # Output channel 2
#     [
#         # Input channel 1
#         [
#             [1, 2],
#             [3, 4]
#         ],
#         # Input channel 2
#         [
#             [2, 3],
#             [4, 5]
#         ]
#     ],
#     # Output channel 3
#     [
#         # Input channel 1
#         [
#             [2, 3],
#             [4, 5]
#         ],
#         # Input channel 2
#         [
#             [3, 4],
#             [5, 6]
#         ]
#     ]
# ])
print(K.shape)
# torch.Size([3, 2, 2, 2])
Y = corr2d_multi_input_output_channels(X, K)
print(Y)
# tensor([[[ 56.,  72.],
#          [104., 120.]],
#         [[ 76., 100.],
#          [148., 172.]],
#         [[ 96., 128.],
#          [192., 224.]]])
print(Y.shape)
# torch.Size([3, 2, 2])
# Explanation:
# X: Input tensor of size c_i * x_h * x_w = 3 * 2 * 2
# K: Kernel tensor of size c_o * c_i * x_h * x_w = 3 * 2 * 2 * 2
# Y: Output tensor of size 3 * 2 * 2


# 1 x 1 Convolutional Layer
# 1 x 1 Convolutional Layer is equivalent to the fully-connected layer, when applied on a per-pixel basis.
# 1 x 1 Convolutional Layer is typically used to adjust the number of channels between network layers and to control
# model complexity.
def corr2d_multi_input_output_channels_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-correlation (convolution) with multiple input and output channels, using 1x1 convolutional layer
    :param X: Input tensor of size x_channels (c_i) * x_h * x_w
    :param K: Kernel tensor of size x_output_channels (c_o) * x_channels (c_i) * k_h (=1)* k_w (=1)
    :return: Y: Output tensor
    """
    c_i, x_h, x_w = X.shape
    c_o = K.shape[0]
    # Input tensor: matrix in each channel dimension is vectorized
    X = X.reshape((c_i, x_h*x_w))
    # 1x1 convolutional layer, only c_o x c_i dimensions can be of dim > 1
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully-connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, x_h, x_w))


# Example
X = torch.normal(mean=0, std=1, size=(3, 3, 3))
print(X.shape)
# torch.Size([3, 3, 3])
K = torch.normal(mean=0, std=1, size=(2, 3, 1, 1))
print(K.shape)
# torch.Size([2, 3, 1, 1])
Y1 = corr2d_multi_input_output_channels_1x1(X, K)
print(Y1.shape)
# torch.Size([2, 3, 3])
Y2 = corr2d_multi_input_output_channels(X, K)
print(Y2.shape)
# torch.Size([2, 3, 3])
assert float(torch.abs(Y1-Y2).sum()) < 1e-6
# No complains, assertion is true


"""
Pooling
"""


# Max pooling and Average pooling
def pool2d(X: torch.Tensor, pool_size: _size_6_t, mode='max'):
    """
    :param X: Input tensor of size x_h * x_w
    :param pool_size: Size of pooling layer
    :param mode: max or avg, depending of pooling layer operation
    :return: Output tensor Y, of size (x_h - p_h + 1) * (x_w - p_w + 1)
    """
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):  # 1st dim: height
        for j in range(Y.shape[1]):  # 2nd dim: width
            if mode == 'max':
                Y[i, j] = X[i: i+p_h, j: j+p_w].max()
            if mode == 'avg':
                Y[i, j] = X[i: i+p_h, j: j+p_w].mean()
    return Y


# Example
X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
])
# Max-pooling layer of size 2*2
print(pool2d(X, (2, 2)))
# tensor([[4., 5.],
#         [7., 8.]])
# Average-pooling layer of size 2*2
print(pool2d(X, (2, 2), mode='avg'))
# tensor([[2., 3.],
#         [5., 6.]])


# Padding and Stride
# Pooling layers can also change the output shape by padding the input and adjusting the stride
# Example
# Input tensor X of size num_examples(batch_size) * num_channels * x_h * x_w = 1 * 1 * 4 * 4
# Number of examples and number of channels are set to 1
X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
print(X)
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]]]])
print(X.shape)
# torch.Size([1, 1, 4, 4])
# Important: By default, the stride and the pooling window in the instance from the framework's built-in
# nn.MaxPool2d class have the same shape. Below, we use pooling window of shape (3, 3), and we get a stride
# shape of (3, 3) by default
pool2d = torch.nn.MaxPool2d(kernel_size=3)
Y = pool2d(X)
print(Y)
# tensor([[[[10.]]]])
print(Y.shape)
# torch.Size([1, 1, 1, 1])

# Stride and padding can be manually specified
pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
Y = pool2d(X)
print(Y)
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])
print(Y.shape)
# torch.Size([1, 1, 2, 2])

# We can also specify arbitrary rectangular pooling window
pool2d = torch.nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1), stride=(2, 3))
Y = pool2d(X)
print(Y)
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])
print(Y.shape)
# torch.Size([1, 1, 2, 2])


# Multiple channels
# When using multi-channel input data, the pooling layer pools each input channel separately.
# As a result, the number of output channels for the pooling layer is the same as the number of input channels.
# Create input tensor X, by concatenating tensors X and X+1 on the channel dimension (1-st dimension)
X = torch.cat((X, X+1), 1)
print(X)
# tensor([
#     # Input channel 1
#     [
#         # Output channel 1
#         [
#             [ 0.,  1.,  2.,  3.],
#             [ 4.,  5.,  6.,  7.],
#             [ 8.,  9., 10., 11.],
#             [12., 13., 14., 15.]
#         ],
#         # Output channel 2
#         [
#             [ 1.,  2.,  3.,  4.],
#             [ 5.,  6.,  7.,  8.],
#             [ 9., 10., 11., 12.],
#             [13., 14., 15., 16.]
#         ]
#     ]
# ])
print(X.shape)
# torch.Size([1, 2, 4, 4])
pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
Y = pool2d(X)
# Note that, the number of output channels is still 2 after pooling
print(Y)
# tensor([
#     # Input channel 1
#     [
#         # Output channel 1
#         [
#             [ 5.,  7.],
#             [13., 15.]
#         ],
#         # Output channel 2
#         [
#             [ 6.,  8.],
#             [14., 16.]
#         ]
#     ]
# ])
print(Y.shape)
# torch.Size([1, 2, 2, 2])


"""
CNNs: LeNet
"""

# LeNet Implementation
net = torch.nn.Sequential(
    # Convolutional layer 1: (1, 1, 28, 28) -> Conv2d, with padding=2 -> (1, 6, 28, 28) -> sigmoid -> (1, 6, 28, 28)
    # -> average pooling -> (1, 16, 14, 14)
    torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
    torch.nn.Sigmoid(),
    torch.nn.AvgPool2d(kernel_size=2, stride=2),
    # Convolutional layer 2: (1, 16, 14, 14) -> Conv2d, without padding -> (1, 16, 10, 10) -> sigmoid -> (1, 16, 10, 10)
    # -> average pooling -> (1, 16, 5, 5)
    torch.nn.Conv2d(6, 16, kernel_size=5),
    torch.nn.Sigmoid(),
    torch.nn.AvgPool2d(kernel_size=2, stride=2),
    # Flatten, convert 4-D tensor to 2-D tensor, before passing it through FC layers
    # (1, 16, 5, 5) -> flatten -> (1, 400)
    torch.nn.Flatten(),
    # 3 Fully-connected layers
    # FC-1: (1, 400) -> (1, 120) -> sigmoid
    torch.nn.Linear(16 * 5 * 5, 120),
    torch.nn.Sigmoid(),
    # FC-2: (1, 120) -> (1, 84) -> sigmoid
    torch.nn.Linear(120, 84),
    torch.nn.Sigmoid(),
    # FC-3: (1, 84) -> (1, 10)
    torch.nn.Linear(84, 10)
)
print(net)
# Sequential(
#     (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (1): Sigmoid()
#     (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#     (4): Sigmoid()
#     (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (6): Flatten(start_dim=1, end_dim=-1)
#     (7): Linear(in_features=400, out_features=120, bias=True)
#     (8): Sigmoid()
#     (9): Linear(in_features=120, out_features=84, bias=True)
#     (10): Sigmoid()
#     (11): Linear(in_features=84, out_features=10, bias=True)
# )

# Test LeNet of image of size 28 * 28, print imtermediate output shapes
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
# Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
# AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
# Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
# Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
# AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
# Flatten output shape: 	 torch.Size([1, 400])
# Linear output shape: 	 torch.Size([1, 120])
# Sigmoid output shape: 	 torch.Size([1, 120])
# Linear output shape: 	 torch.Size([1, 84])
# Sigmoid output shape: 	 torch.Size([1, 84])
# Linear output shape: 	 torch.Size([1, 10])

# Note: Convolutional layers are typically arranged sl that they gradually decrease the spatial resolution of the
# representations, while increasing the number of channels.


# Training
# We will train LeNet on Fashion MNIST dataset
# Example support from these links for loading datasets
# https://boscoj2008.github.io/customCNN/
# https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
# https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
batch_size = 256
transformation = transforms.Compose([transforms.ToTensor()])
data_root_path = "./data"
train_set = torchvision.datasets.FashionMNIST(
    data_root_path,
    download=True,
    train=True,
    transform=transformation
)
print(train_set)
# Dataset FashionMNIST
# Number of datapoints: 60000
# Root location: ./data
# Split: Train
# StandardTransform
# Transform: Compose(
#     ToTensor()
# )
test_set = torchvision.datasets.FashionMNIST(
    data_root_path,
    download=True,
    train=False,
    transform=transformation
)
print(test_set)
# Dataset FashionMNIST
# Number of datapoints: 10000
# Root location: ./data
# Split: Test
# StandardTransform
# Transform: Compose(
#     ToTensor()
# )


from torch.utils.data.dataloader import DataLoader
# Use data loader
train_loader = torch.utils.data.dataloader.DataLoader(train_set, batch_size=batch_size)
print(train_loader)
# <torch.utils.data.dataloader.DataLoader object at 0x10c8137f0>
test_loader = torch.utils.data.dataloader.DataLoader(test_set, batch_size=batch_size)
print(test_loader)
# <torch.utils.data.dataloader.DataLoader object at 0x10c881a60>

# Delete data folder
import shutil
shutil.rmtree(data_root_path)


# Accumulator class borrowed from d2l library
class Accumulator:
    """
    For accumulating sums over `n` variables.
    """
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y): #@save
    """
    Compute the number of correct predictions.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net: torch.nn.Module, data_iter, device=None):  # @save
    """
    Compute accuracy for a model on a dataset using GPU
    :param net:
    :param data_iter:
    :param device:
    :return:
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # Number of correct predictions, number of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT fine-tuning
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """
    Train a model with a GPU (defined in Chapter 6).
    """
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')


# Now train LeNet model on Fashion-MNIST dataset
learning_rate, num_epochs = 0.9, 10
train_ch6(
    net=net,
    train_iter=train_loader,
    test_iter=test_loader,
    num_epochs=num_epochs,
    lr=learning_rate,
    device=None  # torch.device('cuda')
)
