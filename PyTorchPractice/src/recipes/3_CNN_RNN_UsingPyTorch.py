import torch
print(torch.__version__)
# 1.10.0

"""
1. Setting up a loss function
"""
t_c = torch.tensor([1., 2., 3., 4., 5.])
t_u = torch.tensor([5., 4., 3., 2., 1.])


# Define model: Linear model: w * t_u +b
# t_u: Tensor used
# w: Weight tensor
# b: Constant tensor
def model(t_u, w, b):
    return w * t_u + b


# Define loss function
# t_p: Tensor predicted
# t_c: Tensor pre-computed
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# Apply the model and loss
w = torch.ones(1)
b = torch.zeros(1)
t_p = model(t_u, w, b)
print(t_p)
# tensor([5., 4., 3., 2., 1.])
loss = loss_fn(t_p, t_c)
print(loss)
# tensor(8.)
# Explanation: t_p = [5., 4., 3., 2., 1.], t_c = [1., 2., 3., 4., 5.]
# mean([16, 4, 0, 4, 16]) = 8

# Rate of change, loss function, update in SGD
delta = 0.1
learning_rate = 1e-2
# Update w
loss_rate_of_change_w = (loss_fn( model(t_u, w+delta, b), t_c ) -
                         loss_fn( model(t_u, w-delta, b), t_c )) / (2.0 * delta)
w = w - learning_rate * loss_rate_of_change_w
print(w)
# tensor([0.9200])
# Previous value was 1
# Update b
loss_rate_of_change_b = (loss_fn( model(t_u, w, b+delta), t_c ) -
                         loss_fn( model(t_u, w, b-delta), t_c )) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b
print(b)
# tensor([0.0048])
# Previous value was 0

# MSE loss example
from torch import nn
loss = nn.MSELoss()
input = torch.randn(2, 4, requires_grad=True)
target = torch.randn(2, 4)
output = loss(input, target)
output.backward()
print(output)
# tensor(1.9499, grad_fn=<MseLossBackward0>)
print(output.data)
# tensor(1.9499)
print(output.grad_fn)
# <MseLossBackward0 object at 0x10591f190>


"""
2. Estimating the derivative of the loss function
"""


# Derivative of loss function loss_fn defined above
# loss_fn = mean( (t_p - t_c)**2 )
# dloss_fn = 2 * (t_p - t_c)
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs


# Derivative of model function above, w.r.t parameter w
# model = w * t_u + b
# dmodel_dw = t_u
def dmodel_dw(t_u, w, b):
    return t_u


# Derivative of model function above, w.r.t parameter b
# model = w * t_u + b
# dmodel_db = 1
def dmodel_db(t_u, w, b):
    return 1.0


# Derivative of loss function loss_fn, w.r.t parameters w and b
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])


# Parameter estimation iteration method
def parameter_estimation(t_u, params, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        # Forward pass
        w, b = params
        t_p = model(t_u, w, b)
        # Calculate loss
        loss = loss_fn(t_p, t_c)
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        # Backward pass
        grad = grad_fn(t_u, t_c, t_p, w, b)
        # Parameters and gradient
        print("Params: ", params)
        print("Grad: ", grad)
        # Update params
        params = params - learning_rate * grad


# Experiment 1: Baseline parameters
params = torch.tensor([1.0, 0.0])
num_epochs = 10
learning_rate = 1e-2
parameter_estimation(t_u=t_u, params=params, learning_rate=learning_rate, num_epochs=num_epochs)
# Epoch 0, Loss 8.000000
# Params:  tensor([1., 0.])
# Grad:  tensor([8., 0.])
# Epoch 1, Loss 7.430399
# Params:  tensor([0.9200, 0.0000])
# Grad:  tensor([ 6.2400, -0.4800])
# Epoch 2, Loss 7.079777
# Params:  tensor([0.8576, 0.0048])
# Grad:  tensor([ 4.8960, -0.8448])
# Epoch 3, Loss 6.856889
# Params:  tensor([0.8086, 0.0132])
# Grad:  tensor([ 3.8696, -1.1217])
# Epoch 4, Loss 6.708566
# Params:  tensor([0.7699, 0.0245])
# Grad:  tensor([ 3.0856, -1.3314])
# Epoch 5, Loss 6.603817
# Params:  tensor([0.7391, 0.0378])
# Grad:  tensor([ 2.4866, -1.4899])
# Epoch 6, Loss 6.524587
# Params:  tensor([0.7142, 0.0527])
# Grad:  tensor([ 2.0290, -1.6093])
# Epoch 7, Loss 6.460349
# Params:  tensor([0.6939, 0.0688])
# Grad:  tensor([ 1.6791, -1.6989])
# Epoch 8, Loss 6.404971
# Params:  tensor([0.6771, 0.0858])
# Grad:  tensor([ 1.4117, -1.7656])
# Epoch 9, Loss 6.354877
# Params:  tensor([0.6630, 0.1034])
# Grad:  tensor([ 1.2070, -1.8150])

# Experiment 2: Reduce learning rate
# Loss reduction is slower
learning_rate = 1e-4
parameter_estimation(t_u=t_u, params=params, learning_rate=learning_rate, num_epochs=num_epochs)
# Epoch 0, Loss 8.000000
# Params:  tensor([1., 0.])
# Grad:  tensor([8., 0.])
# Epoch 1, Loss 7.993607
# Params:  tensor([0.9992, 0.0000])
# Grad:  tensor([ 7.9824e+00, -4.8001e-03])
# Epoch 2, Loss 7.987242
# Params:  tensor([9.9840e-01, 4.8001e-07])
# Grad:  tensor([ 7.9648, -0.0096])
# Epoch 3, Loss 7.980905
# Params:  tensor([9.9761e-01, 1.4389e-06])
# Grad:  tensor([ 7.9473, -0.0144])
# Epoch 4, Loss 7.974596
# Params:  tensor([9.9681e-01, 2.8754e-06])
# Grad:  tensor([ 7.9298, -0.0191])
# Epoch 5, Loss 7.968314
# Params:  tensor([9.9602e-01, 4.7885e-06])
# Grad:  tensor([ 7.9124, -0.0239])
# Epoch 6, Loss 7.962060
# Params:  tensor([9.9523e-01, 7.1770e-06])
# Grad:  tensor([ 7.8950, -0.0286])
# Epoch 7, Loss 7.955835
# Params:  tensor([9.9444e-01, 1.0040e-05])
# Grad:  tensor([ 7.8777, -0.0334])
# Epoch 8, Loss 7.949636
# Params:  tensor([9.9365e-01, 1.3376e-05])
# Grad:  tensor([ 7.8604, -0.0381])
# Epoch 9, Loss 7.943464
# Params:  tensor([9.9286e-01, 1.7184e-05])
# Grad:  tensor([ 7.8431, -0.0428])


# Experiment 3: Change tensor scale / norm
# Compare Experiment 3 vs 1: Loss starts higher, drops faster
t_un = 0.1 * t_u
learning_rate = 1e-2
parameter_estimation(t_u=t_un, params=params, learning_rate=learning_rate, num_epochs=num_epochs)
# Epoch 0, Loss 9.710000
# Params:  tensor([1., 0.])
# Grad:  tensor([-1.1800, -5.4000])
# Epoch 1, Loss 9.407789
# Params:  tensor([1.0118, 0.0540])
# Grad:  tensor([-1.1450, -5.2849])
# Epoch 2, Loss 9.118546
# Params:  tensor([1.0233, 0.1068])
# Grad:  tensor([-1.1108, -5.1724])
# Epoch 3, Loss 8.841709
# Params:  tensor([1.0344, 0.1586])
# Grad:  tensor([-1.0773, -5.0622])
# Epoch 4, Loss 8.576743
# Params:  tensor([1.0451, 0.2092])
# Grad:  tensor([-1.0446, -4.9545])
# Epoch 5, Loss 8.323136
# Params:  tensor([1.0556, 0.2587])
# Grad:  tensor([-1.0125, -4.8492])
# Epoch 6, Loss 8.080397
# Params:  tensor([1.0657, 0.3072])
# Grad:  tensor([-0.9812, -4.7461])
# Epoch 7, Loss 7.848055
# Params:  tensor([1.0755, 0.3547])
# Grad:  tensor([-0.9506, -4.6453])
# Epoch 8, Loss 7.625663
# Params:  tensor([1.0850, 0.4011])
# Grad:  tensor([-0.9206, -4.5467])
# Epoch 9, Loss 7.412791
# Params:  tensor([1.0942, 0.4466])
# Grad:  tensor([-0.8913, -4.4502])


# Experiment 4: Increase number of epochs from 10 to 1000
# Compare Experiment 4 vs 1: Final loss value is lower, loss reaches convergence
# Final parameters are: [-0.9428,  5.7934]
num_epochs = 1000
parameter_estimation(t_u=t_u, params=params, learning_rate=learning_rate, num_epochs=num_epochs)
# Last 10 epochs only
# ...
# Epoch 990, Loss 0.008255
# Params:  tensor([-0.9410,  5.7870])
# Grad:  tensor([ 0.0199, -0.0720])
# Epoch 991, Loss 0.008199
# Params:  tensor([-0.9412,  5.7878])
# Grad:  tensor([ 0.0199, -0.0718])
# Epoch 992, Loss 0.008143
# Params:  tensor([-0.9414,  5.7885])
# Grad:  tensor([ 0.0198, -0.0715])
# Epoch 993, Loss 0.008088
# Params:  tensor([-0.9416,  5.7892])
# Grad:  tensor([ 0.0197, -0.0713])
# Epoch 994, Loss 0.008034
# Params:  tensor([-0.9418,  5.7899])
# Grad:  tensor([ 0.0197, -0.0710])
# Epoch 995, Loss 0.007980
# Params:  tensor([-0.9420,  5.7906])
# Grad:  tensor([ 0.0196, -0.0708])
# Epoch 996, Loss 0.007926
# Params:  tensor([-0.9422,  5.7913])
# Grad:  tensor([ 0.0195, -0.0705])
# Epoch 997, Loss 0.007872
# Params:  tensor([-0.9424,  5.7920])
# Grad:  tensor([ 0.0195, -0.0703])
# Epoch 998, Loss 0.007819
# Params:  tensor([-0.9426,  5.7927])
# Grad:  tensor([ 0.0194, -0.0701])
# Epoch 999, Loss 0.007766
# Params:  tensor([-0.9428,  5.7934])
# Grad:  tensor([ 0.0193, -0.0698])


"""
3. Fine-tuning a model
"""
# How to find the gradients of the loss function by applying an optimization function?
# We will use the backward() function

# First, reset parameters
if params.grad is not None:
    params.grad.zero_()


# Parameter estimation iteration method using backward() method
# Uses built-in backpropagation method
def parameter_estimation_using_backward_method(t_u, params, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        # Forward pass
        t_p = model(t_u, *params)
        # Calculate loss
        loss = loss_fn(t_p, t_c)
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        # Backward pass
        if params.grad is not None:
            params.grad.zero_()
        loss.backward()
        # Parameters and gradient
        print("Params: ", params)
        # Update params
        params = (params - learning_rate * params.grad).detach().requires_grad_()


# Fine-tune model using backward() method
# Final parameters are: [-0.9428,  5.7934] (same as parameter_estimation() method output)
params = torch.tensor([1.0, 0.0], requires_grad=True)
num_epochs = 1000
learning_rate = 1e-2
parameter_estimation_using_backward_method(t_u=t_u, params=params, learning_rate=learning_rate, num_epochs=num_epochs)
# Last 10 epochs only
# ...
# Epoch 990, Loss 0.008255
# Params:  tensor([-0.9410,  5.7870], requires_grad=True)
# Epoch 991, Loss 0.008199
# Params:  tensor([-0.9412,  5.7878], requires_grad=True)
# Epoch 992, Loss 0.008143
# Params:  tensor([-0.9414,  5.7885], requires_grad=True)
# Epoch 993, Loss 0.008088
# Params:  tensor([-0.9416,  5.7892], requires_grad=True)
# Epoch 994, Loss 0.008034
# Params:  tensor([-0.9418,  5.7899], requires_grad=True)
# Epoch 995, Loss 0.007980
# Params:  tensor([-0.9420,  5.7906], requires_grad=True)
# Epoch 996, Loss 0.007926
# Params:  tensor([-0.9422,  5.7913], requires_grad=True)
# Epoch 997, Loss 0.007872
# Params:  tensor([-0.9424,  5.7920], requires_grad=True)
# Epoch 998, Loss 0.007819
# Params:  tensor([-0.9426,  5.7927], requires_grad=True)
# Epoch 999, Loss 0.007766
# Params:  tensor([-0.9428,  5.7934], requires_grad=True)


"""
4. Selecting an Optimization Function
"""
import torch.optim as optim
print(dir(optim))
# ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 'Optimizer', 'RAdam', 'RMSprop', 'Rprop',
# 'SGD', 'SparseAdam', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__',
# '__path__', '__spec__', '_functional', '_multi_tensor', 'lr_scheduler', 'swa_utils']

# Define SGD optimizer
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)
print(optimizer)
# SGD (
# Parameter Group 0
#     dampening: 0
#     lr: 1e-05
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )

# Use SGD optimizer
t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()
optimizer.step()
print(params)
# tensor([0.9999, 0.0000], requires_grad=True)

# Define and use new SGD optimizer with lower learning rate
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
# Set the gradient of all optimized tensors to zero
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(params)
# tensor([0.9200, 0.0000], requires_grad=True)


# Parameter estimation iteration method using backward() method
# Uses built-in backpropagation method
def parameter_estimation_using_optimizer_backward(t_u, params, num_epochs, optimizer):
    for epoch in range(num_epochs):
        # Forward pass
        t_p = model(t_u, *params)
        # Calculate loss
        loss = loss_fn(t_p, t_c)
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t_p = model(t_u, *params)
    # Parameters and gradient
    print("Params: ", params)


# Experiment 1: SGD Optimizer
num_epochs = 1000
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
parameter_estimation_using_optimizer_backward(t_u=t_u, params=params, num_epochs=num_epochs, optimizer=optimizer)
# Last 10 only
# ...
# Epoch 990, Loss 0.008199
# Epoch 991, Loss 0.008143
# Epoch 992, Loss 0.008088
# Epoch 993, Loss 0.008034
# Epoch 994, Loss 0.007980
# Epoch 995, Loss 0.007926
# Epoch 996, Loss 0.007872
# Epoch 997, Loss 0.007819
# Epoch 998, Loss 0.007766
# Epoch 999, Loss 0.007714
# Params:  tensor([-0.9432,  5.7948], requires_grad=True)


# Experiment 2: Adam Optimizer
# Result: Adam optimizer converges loss to 0 faster (after Epoch 86, the loss is 0.
num_epochs = 1000
learning_rate = 1e-2
optimizer = optim.Adam([params], lr=learning_rate)
parameter_estimation_using_optimizer_backward(t_u=t_u, params=params, num_epochs=num_epochs, optimizer=optimizer)
# Last 10 only
# ...
# Epoch 990, Loss 0.000000
# Epoch 991, Loss 0.000000
# Epoch 992, Loss 0.000000
# Epoch 993, Loss 0.000000
# Epoch 994, Loss 0.000000
# Epoch 995, Loss 0.000000
# Epoch 996, Loss 0.000000
# Epoch 997, Loss 0.000000
# Epoch 998, Loss 0.000000
# Epoch 999, Loss 0.000000
# Params:  tensor([-1.,  6.], requires_grad=True)


"""
5. Further Optimizing the Loss Function
"""
# How to optimize the training set, and test it with a validation set using random samples

# Use 20% of data as validation set, using shuffled_indices
n_samples = t_u.shape[0]
print(n_samples)
# 5
n_val = int(0.4 * n_samples)
# Shuffle indices
shuffled_indices = torch.randperm(n_samples)
print(shuffled_indices)
# tensor([2, 4, 0, 1, 3])
# Define indices for training and validation sets
train_indices = shuffled_indices[:-n_val]
validation_indices = shuffled_indices[-n_val:]
print(train_indices)
# tensor([2, 4, 0])
print(validation_indices)
# tensor([1, 3])
# Define training / validation sets for t_u and t_c
t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]
print(t_u_train)
# tensor([3., 1., 5.])
print(t_c_train)
# tensor([3., 5., 1.])
t_u_validation = t_u[validation_indices]
t_c_validation = t_c[validation_indices]
print(t_u_validation)
# tensor([4., 2.])
print(t_c_validation)
# tensor([2., 4.])


# Parameters
params = torch.tensor([1.0, 0.0], requires_grad=True)
num_epochs = 1000
learning_rate = 1e-2
optimizer = optim.SGD([params], learning_rate)
# Define t_un_train and t_un_validation
t_un_train = 0.1 * t_u_train
t_un_validation = 0.1 * t_u_validation
print(t_un_train)
# tensor([0.3000, 0.1000, 0.5000])
print(t_un_validation)
# tensor([0.4000, 0.2000])

# CAREFUL: t_p is attached to gradient from a previous op, detach it
# print(t_p)
# tensor([5., 4., 3., 2., 1.], grad_fn=<AddBackward0>)
# t_p = t_p.detach()
# print(t_p)
# tensor([5., 4., 3., 2., 1.])

loss = loss_fn(t_p, t_c)

# Parameter estimation using training / validation sets
def parameter_estimation_using_training_validation_sets():
    for epoch in range(num_epochs):
        # 1. Forward pass
        # 1.1. Training set
        t_p_train = model(t_un_train, *params)
        loss_train = loss_fn(t_p_train, t_c_train)
        # 1.2. Validation set
        with torch.no_grad():
            t_p_validation = model(t_un_validation, *params)
            loss_validation = loss_fn(t_p_validation, t_c_validation)
        # t_p_validation = model(t_un_validation, *params)
        # loss_validation = loss_fn(t_p_validation, t_c_validation)
        # Print losses
        print('Epoch %d, Training Loss %f, Validation loss %f' % (epoch, float(loss_train), float(loss_validation)))
        # t_p_validation.detach_()
        # Backward pass
        print("Before zero_grad call")
        optimizer.zero_grad()
        print("Before backward call")
        loss.backward(retain_graph=True)
        # loss.backward()
        print("Before step call")
        optimizer.step()
    t_p = model(t_un, *params)
    print("Params: ", params)


# Experiment 1
# TODO
#  Uncomment after fixing the error that occurs when calling backward() method
#  Error message: "RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors
#  after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or
#  autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to
#  access saved tensors after calling backward."
# parameter_estimation_using_training_validation_sets()
