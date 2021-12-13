"""
1. Sampling Tensors
"""
# Use case: Weight initialization in DNNs / CNNs / RNNs using probability distributions

import torch

# Use manual seed, to be able to reproduce the same set of results
print(torch.manual_seed(1234))
# <torch._C.Generator object at 0x102d01b70>
torch.manual_seed(1234)

# Random Normal distribution
print(torch.randn(4, 4))
# tensor([[-0.1117, -0.4966,  0.1631, -0.8817],
#         [ 0.0539,  0.6684, -0.0597, -0.4675],
#         [-0.2153,  0.8840, -0.7584, -0.3689],
#         [-0.3424, -1.4020,  0.3206, -1.0219]])

# Continuous uniform distribution
# f(x) = {
#   1 / (b-a)   ; for a <= x <= b
#   0           ; for x < a or x > b
# }
print(torch.Tensor(4, 4))
# tensor([[-0.1113, -0.4966,  0.1631, -0.8817],
#         [ 0.0539,  0.6684, -0.0597, -0.4675],
#         [-0.2153,  0.8840, -0.7584, -0.3689],
#         [-0.3424, -1.4020,  0.3206, -1.0219]])
# Random number from uniform distribution U(0, 1)
print(torch.Tensor(4, 4).uniform_(0, 1))
# tensor([[0.2837, 0.6567, 0.2388, 0.7313],
#         [0.6012, 0.3043, 0.2548, 0.6294],
#         [0.9665, 0.7399, 0.4517, 0.4757],
#         [0.7842, 0.1525, 0.6662, 0.3343]])

# Bernoulli distribution: Probability mass function
# q = {
#     1-p   ; for k=0
#     p     ; for k=1
# }
torch.bernoulli(torch.Tensor(4, 4).uniform_(0, 1))
# tensor([[0., 0., 0., 0.],
#         [1., 0., 1., 0.],
#         [1., 0., 1., 1.],
#         [0., 0., 0., 0.]])

# Multinomial distribution
# Wikipedia: https://en.wikipedia.org/wiki/Multinomial_distribution
#   PMF: \frac{n!}{x_1!\cdots x_k!} p_1^{x_1} \cdots p_k^{x_k}
# torch.multinomial: https://pytorch.org/docs/stable/generated/torch.multinomial.html
torch.multinomial(torch.tensor([
    10., 10., 13., 10.,
    34., 45., 65., 67.,
    87., 89., 87., 34.
]), 3)
# tensor([4, 5, 7])
# Multinomial distribution with replacement
torch.multinomial(torch.tensor([
    10., 10., 13., 10.,
    34., 45., 65., 67.,
    87., 89., 87., 34.
]), 3, replacement=True)
# tensor([10,  4,  9])

# Normal distribution
# Weight initialization using Normal distribution: Commonly used in DNNs, RNNs, CNNs.
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
# tensor([1.8971, 1.9295, 3.4439, 3.9409, 5.3542, 5.4984, 6.2851, 8.0161, 9.1649, 9.9428])
torch.normal(mean=0.5, std=torch.arange(1., 6.))
# tensor([ 0.0124,  0.8888, -2.6443,  0.2037, -1.2247])
torch.normal(mean=0.5, std=torch.arange(0.2, 0.6))
# tensor([0.1911])


"""
2. Variable Tensors
"""
from torch.autograd import Variable
print(Variable(torch.ones(2, 2), requires_grad=True))
# tensor([[1., 1.],
#        [1., 1.]], requires_grad=True)
a, b = 2, 3
x1 = Variable(torch.randn(a, b), requires_grad=True)
x2 = Variable(torch.randn(a, b), requires_grad=True)
x3 = Variable(torch.randn(a, b), requires_grad=True)
c = x1 * x2
print(c)
# tensor([[-1.3006, -0.7797,  0.7862],
#         [ 0.3897, -0.2941, -0.0319]], grad_fn=<MulBackward0>)
d = a + x3
print(d)
# tensor([[3.6285, 2.9121, 2.7460],
#         [1.1008, 1.2243, 2.2259]], grad_fn=<AddBackward0>)
e = torch.sum(d)
print(e)
# tensor(13.8376, grad_fn=<SumBackward0>)
e.backward()
print(e)
# tensor(13.8376, grad_fn=<SumBackward0>)


"""
3. Basic Statistics
"""
# For 1-D, mean / median / mode is simple
# For n-D with n > 1, the dimension needs to be specified
# Mean, Median, Mode
aTensor = torch.tensor([1., 2., 3., 4.])
print(torch.mean(aTensor))
# tensor(2.5000)
print(torch.median(aTensor))
# tensor(2.)
print(torch.mode(aTensor))
# torch.return_types.mode(values=tensor(1.),indices=tensor(0))
# 2-D mean, median, mode
d = torch.randn(2, 3)
print(d)
# tensor([[ 0.8265,  1.0747, -0.7912],
#         [ 0.0673,  0.3375, -0.4594]])
# Mean of 2-D tensor across different dimensions
print(torch.mean(d, dim=0))
# tensor([ 0.4469,  0.7061, -0.6253])
print(torch.mean(d, dim=1))
# tensor([ 0.3700, -0.0182])
# Median
print(torch.median(d, dim=0))
# torch.return_types.median(values=tensor([ 0.0673,  0.3375, -0.7912]), indices=tensor([1, 1, 0]))
print(torch.median(d, dim=1))
# torch.return_types.median(values=tensor([0.8265, 0.0673]), indices=tensor([0, 0]))
print(torch.mode(d, dim=0))
# torch.return_types.mode(values=tensor([ 0.0673,  0.3375, -0.7912]), indices=tensor([1, 1, 0]))
print(torch.mode(d, dim=1))
# torch.return_types.mode(values=tensor([-0.7912, -0.4594]), indices=tensor([2, 2]))
# Std
print(torch.std(d))
# tensor(0.7222)
print(torch.std(d, dim=0))
# tensor([0.5368, 0.5213, 0.2346])
print(torch.std(d, dim=1))
# tensor([1.0133, 0.4053])
# Var
print(torch.var(d))
# tensor(0.5216)
print(torch.var(d, dim=0))
# tensor([0.2882, 0.2717, 0.0551])
print(torch.var(d, dim=1))
# tensor([1.0267, 0.1642])

"""
4. Gradient Computation
"""
# Using forward pass (Variable w will be defined later)


def forward(x):
    return x * w


x_data = [11., 22., 33.]
y_data = [21., 14., 64.]
w = Variable(torch.Tensor([1.0]), requires_grad=True)
# Before training
print("Predict (before training):", 4, forward(4).data[0])
# Predict (before training): 4 tensor(4.)


# Define loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Run the training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        # Manually set the gradients to zero after updating weights
        w.grad.data.zero_()
    print("progress: ", epoch, l.data[0])
#     grad:  11.0 21.0 tensor(-220.)
#     grad:  22.0 14.0 tensor(2481.6001)
#     grad:  33.0 64.0 tensor(-51303.6484)
# progress:  0 tensor(604238.8125)
#     grad:  11.0 21.0 tensor(118461.7578)
#     grad:  22.0 14.0 tensor(-671630.6875)
#     grad:  33.0 64.0 tensor(13114108.)
# progress:  1 tensor(3.9481e+10)
#     grad:  11.0 21.0 tensor(-30279010.)
#     grad:  22.0 14.0 tensor(1.7199e+08)
#     grad:  33.0 64.0 tensor(-3.3589e+09)
# progress:  2 tensor(2.5900e+15)
#     grad:  11.0 21.0 tensor(7.7553e+09)
#     grad:  22.0 14.0 tensor(-4.4050e+10)
#     grad:  33.0 64.0 tensor(8.6030e+11)
# progress:  3 tensor(1.6991e+20)
#     grad:  11.0 21.0 tensor(-1.9863e+12)
#     grad:  22.0 14.0 tensor(1.1282e+13)
#     grad:  33.0 64.0 tensor(-2.2034e+14)
# progress:  4 tensor(1.1146e+25)
#     grad:  11.0 21.0 tensor(5.0875e+14)
#     grad:  22.0 14.0 tensor(-2.8897e+15)
#     grad:  33.0 64.0 tensor(5.6436e+16)
# progress:  5 tensor(7.3118e+29)
#     grad:  11.0 21.0 tensor(-1.3030e+17)
#     grad:  22.0 14.0 tensor(7.4013e+17)
#     grad:  33.0 64.0 tensor(-1.4455e+19)
# progress:  6 tensor(4.7966e+34)
#     grad:  11.0 21.0 tensor(3.3374e+19)
#     grad:  22.0 14.0 tensor(-1.8957e+20)
#     grad:  33.0 64.0 tensor(3.7022e+21)
# progress:  7 tensor(inf)
#     grad:  11.0 21.0 tensor(-8.5480e+21)
#     grad:  22.0 14.0 tensor(4.8553e+22)
#     grad:  33.0 64.0 tensor(-9.4824e+23)
# progress:  8 tensor(inf)
#     grad:  11.0 21.0 tensor(2.1894e+24)
#     grad:  22.0 14.0 tensor(-1.2436e+25)
#     grad:  33.0 64.0 tensor(2.4287e+26)
# progress:  9 tensor(inf)

# After training
print("Predict (after training): ", 4, forward(4).data[0])
# Predict (after training):  4 tensor(-9.2687e+24)

# Compute gradients from a loss function using variable method on the tensor
from torch import FloatTensor
from torch.autograd import Variable

a = Variable(FloatTensor([5]))
weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (12, 53, 91, 73)]
w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
Loss = (10-d)
Loss.backward()
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to Loss: {gradient}")
# Gradient of w1 w.r.t to Loss: -455.0
# Gradient of w2 w.r.t to Loss: -365.0
# Gradient of w3 w.r.t to Loss: -60.0
# Gradient of w4 w.r.t to Loss: -265.0

"""
5. Tensor Operations: Matrix multiplication
"""
# Tensors are wrapped within the Variable, which has 3 properties: grad, volatile, gradient
x = Variable(torch.Tensor(4, 4).uniform_(-4, 5))
y = Variable(torch.Tensor(4, 4).uniform_(-3, 2))
# Matrix multiplication
z = torch.mm(x, y)
print(z.size())
# torch.Size([4, 4])
print(z)
# tensor([[-20.2517,   8.6873,   3.0253,  -8.9543],
#         [-23.1440,   3.8816,  -0.4090, -12.2528],
#         [-18.9649,   1.9516,  -1.1531, -13.6305],
#         [-13.0874,   4.3159,   2.2368,  -5.4833]])

# Properties of Variable
print("Requires Gradient: %s " % (z.requires_grad))
# Requires Gradient: False
print("Volatile: %s " % (z.volatile))
# Volatile: False
print("Gradient: %s " % (z.grad))
# Gradient: None <input>:2: UserWarning: volatile was removed (Variable.volatile is always False)
print(z.data)
# tensor([[-20.2517,   8.6873,   3.0253,  -8.9543],
#         [-23.1440,   3.8816,  -0.4090, -12.2528],
#         [-18.9649,   1.9516,  -1.1531, -13.6305],
#         [-13.0874,   4.3159,   2.2368,  -5.4833]])

"""
6. Tensor Operations: Matrix-Vector, Matrix-Matrix, Vector-Vector Computation
"""
mat1 = torch.FloatTensor(4, 4).uniform_(0, 1)
print(mat1)
# tensor([[0.1651, 0.6424, 0.5656, 0.1501],
#         [0.0887, 0.1748, 0.2352, 0.4894],
#         [0.9073, 0.6679, 0.5323, 0.2173],
#         [0.0214, 0.6277, 0.0261, 0.8818]])
mat2 = torch.FloatTensor(5, 4).uniform_(0, 1)
print(mat2)
# tensor([[0.2524, 0.6175, 0.5152, 0.0832],
#         [0.1967, 0.6081, 0.5845, 0.2562],
#         [0.5316, 0.1124, 0.7639, 0.2386],
#         [0.7486, 0.1226, 0.7588, 0.8689],
#         [0.7542, 0.8851, 0.9451, 0.8503]])
vec1 = torch.FloatTensor(4).uniform_(0, 1)
print(vec1)
# tensor([0.5311, 0.8804, 0.4009, 0.6110])
# Matrix-scalar addition
print(mat1 + 10.5)
# tensor([[10.6651, 11.1424, 11.0656, 10.6502],
#         [10.5887, 10.6748, 10.7352, 10.9894],
#         [11.4073, 11.1679, 11.0323, 10.7173],
#         [10.5214, 11.1277, 10.5261, 11.3818]])
# Vector-matrix addition
print(mat1 + vec1)
# tensor([[0.6962, 1.5228, 0.9665, 0.7612],
#         [0.6199, 1.0552, 0.6361, 1.1005],
#         [1.4385, 1.5483, 0.9333, 0.8283],
#         [0.5525, 1.5080, 0.4270, 1.4928]])

# Matrix-matrix addition: Size mismatch, throws an error
try:
    mat1 + mat2
except RuntimeError:  # Can have multiple exception types
    print("Tensor sizes do not match")
    pass
# Tensor sizes do not match

# Matrix-matrix multiplication: Sizes match
print(mat1 * mat1)
# tensor([[2.7251e-02, 4.1271e-01, 3.1993e-01, 2.2545e-02],
#         [7.8723e-03, 3.0566e-02, 5.5313e-02, 2.3954e-01],
#         [8.2323e-01, 4.4606e-01, 2.8339e-01, 4.7223e-02],
#         [4.5725e-04, 3.9397e-01, 6.8299e-04, 7.7753e-01]])

"""
7. Distributions
"""

# Bernoulli distribution
# Discrete probability distribution of R.V., which takes value 1 when probability of event is success, and 0 otherwise
from torch.distributions.bernoulli import Bernoulli
# Creates Bernoulli distribution parameterized by probs
# Samples are binary (0 or 1). They take the value 1 with probability p and 0 with probability 1-p.
dist = Bernoulli(torch.tensor([0.3, 0.6, 0.9]))
# .sample() is binary: it takes 1 with p and 0 with 1-p
print(dist.sample())
# tensor([0., 1., 1.])

# Beta distribution
# Family of continuous R.V. defined in the range of 0 and 1
from torch.distributions.beta import Beta
dist = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
print(dist)
# Beta()
print(dist.sample())
# tensor([0.9214])

# Binomial distribution
# Family of discrete probability distribution, where probability of success is defined as 1 and failure is 0.
# Binomial distribution models the number of successful events over many trials
from torch.distributions.binomial import Binomial
dist = Binomial(100, torch.tensor([0, 0.2, 0.8, 1]))
print(dist)
# mial(total_count: torch.Size([4]), probs: torch.Size([4]))
print(dist.sample())
# tensor([  0.,  19.,  79., 100.])

# Categorical distribution
# Defined as Generalized Bernouli distribution, which is a discrete probability distribution that explains
# the possible results of any random variable that may take on one of the possible categories, with the
# probability of each category specified in the tensor
from torch.distributions.categorical import Categorical
dist = Categorical(torch.tensor([0.2, 0.2, 0.2, 0.2]))
print(dist)
# Categorical(probs: torch.Size([4]))
print(dist.sample())
# tensor(3)

# Laplacian distribution
# Continuous probability distribution function, also known as double exponential distribution
# Wikipedia: https://en.wikipedia.org/wiki/Laplace_distribution
from torch.distributions.laplace import Laplace
dist = Laplace(torch.tensor([10.0]), torch.tensor([0.99]))
print(dist)
# Laplace(loc: tensor([10.]), scale: tensor([0.9900]))
print(dist.sample())
# tensor([10.1471])

# Normal distribution
# Parameterized by loc and scale
# [mu +/- (1*std)] => 68.26%
# [mu +/- (2*std)] => 95.44%
# [mu +/- (3*std)] => 99.73%
from torch.distributions.normal import Normal
dist = Normal(torch.tensor([100.0]), torch.tensor([10.0]))
print(dist)
# Normal(loc: tensor([100.]), scale: tensor([10.]))
print(dist.sample())
# tensor([102.1055])
