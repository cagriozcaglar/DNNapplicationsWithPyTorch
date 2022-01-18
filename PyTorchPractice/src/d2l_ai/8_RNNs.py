import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
import shutil


def plot_and_show_using_matplotlib(x_values, y_values, x_label, y_label, plot_drawing='b-', show=True):
    """
    Plot using matplotlib, optionally show the plot.
    This method is needed, because d2l.plot is designed to be used with ipython
    :param x_values:
    :param y_values:
    :param plot_drawing:
    :param x_label:
    :param y_label:
    :param show:
    :return:
    """
    plt.plot(x_values, y_values, plot_drawing)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show:
        plt.show()


"""
Sequence Models
"""
# Generate synthetic data using sine function with additive noise for time steps t=1,2,...,1000
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(input=0.01 * time) + torch.normal(mean=0, std=0.2, size=(T,))
# Plot
plt.plot(time, x, 'b-')
plt.xlabel('time')
plt.ylabel('x')
# plt.show()

# Training with first 600 (feature, label) pairs
tau = 4
features = torch.zeros(T - tau, tau)
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
# Only first n_train examples are used for training
train_iter = d2l.load_array(
    data_arrays=(features[:n_train], labels[:n_train]),
    batch_size=batch_size,
    is_train=True
)


# Training


def init_weights(m):
    """
    Initialize the weights of the network
    :param m:
    :return:
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def get_net():
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net


# Note: MSELoss computes squared error without the 1/2 factor
loss = torch.nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    """
    Model training with RNN
    :param net:
    :param train_iter:
    :param loss:
    :param epochs:
    :param lr:
    :return:
    """
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)
# epoch 1, loss: 0.064319
# epoch 2, loss: 0.058503
# epoch 3, loss: 0.055046
# epoch 4, loss: 0.053809
# epoch 5, loss: 0.051242

# Prediction
# One-step-ahead prediction
one_step_preds = net(features)
plot = d2l.plot([time, time[tau:]], [x.detach().numpy(), one_step_preds.detach().numpy()],
                'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
# TODO
#  (Optional) Fix plotting with multiple lines
# plot_and_show_using_matplotlib(
#    x_values=[time, time[tau:]],
#    y_values=[x.detach().numpy(), one_step_preds.detach().numpy()],
#    x_label='time',
#    y_label='x'
# )

# k-step-ahead prediction
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), one_step_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

# k-step-ahead prediction with varying k values = 1, 4, 16, 64
# As k increases, the errors accumulate and the quality of the prediction degrades
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))

"""
Text Preprocessing
"""
import collections
import re

# Reading the dataset
# @save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():  # @save
    """
    Load the time machine dataset into a list of text lines.
    """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def delete_data_folder():
    # Delete generated data folder
    data_root_path = "../data"
    shutil.rmtree(data_root_path)


lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])

# Delete generated data folder
delete_data_folder()

# import os
# print(os.getcwd())
# /Users/cagri/Desktop/Projects/DNNapplicationsWithPyTorch/PyTorchPractice/src/d2l_ai


# Tokenization


def tokenize(lines, token='word'):  # @save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
# ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# []
# []
# []
# []
# ['i']
# []
# []
# ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
# ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
# ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
print(len(tokens))


# 3221


# Vocabulary

class Vocab:  # @save
    """
    Vocabulary for text
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies (in decreasing order of frequency, from highest to lowest)
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The index for the unknown token is 0
        self.index_to_token = ['<unk>'] + reserved_tokens
        self.token_to_index = {token: index for index, token in enumerate(self.index_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_index:
                self.index_to_token.append(token)
                self.token_to_index[token] = len(self.index_to_token) - 1

    def __len__(self):
        return len(self.index_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_index.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.index_to_token[indices]
        return [self.index_to_token[index] for index in indices]

    @property
    def unk(self):   # Index for the unknown token
        return 0

    @property
    def token_freqs(self):   # Token frequencies
        return self._token_freqs


def count_corpus(tokens):  # @save
    """
    Count token frequencies
    :param tokens: 1D list of 2D list of tokens
    :return:
    """
    # If tokens is empty or is a 2-D list of tokens
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# Construct vocabulary
vocab = Vocab(tokens)
print(list(vocab.token_to_index.items())[:10])
# [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]
# Unknown, followed by 9 most frequent words

# Convert text lines to indices
for i in [0, 10]:
    print('words: ', tokens[i])
    print('indices: ', vocab[tokens[i]])
# words:  ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# indices:  [1, 19, 50, 40, 2183, 2184, 400]
# words:  ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
# indices:  [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]


# Putting everything together


def load_corpus_time_machine(max_tokens=-1):   #@save
    """
    Return token indices and the vocabulary of the time machine dataset
    :param max_tokens:
    :return:
    """
    # Read time machine dataset
    lines = read_time_machine()
    # Tokenize text into characters (not words)
    tokens = tokenize(lines, 'char')
    # Generate vocabulary from tokens
    vocab = Vocab(tokens)
    # Flatten all text into one list of characters, the corpus
    corpus = [vocab[token] for line in tokens for token in line]
    # Limit number of tokens if there is a limit
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus))  # Number of characters in the corpus
print(len(vocab))   # Number of distinct characters
# 170580
# 28


"""
Language Models and the Dataset
"""
import random
tokens = tokenize(read_time_machine())
# Concatenate all text lines
corpus = [token for line in tokens for token in line]
# Generate vocabulary from tokens (uses distinct tokens, assigns to vocabulary, plus <unk> and reserved tokens
vocab = Vocab(corpus)
vocab.__len__()
# 4580
print(vocab.token_freqs[:10])
# [('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552),
# ('in', 541), ('that', 443), ('my', 440)]
# Note: Top 10 words above are ** stop words **. We will not remove them, they carry meaning in sequences, which is
# not necessarily the case in bag-of-words approaches

# Unigram word frequencies follow power law / Zipf's law
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
# n_i & 1 / i^a
# => n_i  = c * (1 / i^a)               => Linear dependency
# => log(n_i)  = log(c * (1 / i^a))     => Took log
# => log(n_i)  = log(c) + log(i^(-a))   => Took log
# => log(n_i)  = -a*log(i) + c          => log(c) converted to constant c
# Linear relationship in log-log scale between index i and word frequency n_i => Zipf's law

# Generate bigrams
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])
# [(('of', 'the'), 309), (('in', 'the'), 169), (('i', 'had'), 130), (('i', 'was'), 112), (('and', 'the'), 109),
# (('the', 'time'), 102), (('it', 'was'), 99), (('to', 'the'), 85), (('as', 'i'), 78), (('of', 'a'), 73)]
# Note: Out of top 10 words, 9 of them are composed of stop words (the odd one out is "the time").

# Generate trigrams
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
# [(('the', 'time', 'traveller'), 59), (('the', 'time', 'machine'), 30), (('the', 'medical', 'man'), 24),
# (('it', 'seemed', 'to'), 16), (('it', 'was', 'a'), 15), (('here', 'and', 'there'), 15),
# (('seemed', 'to', 'me'), 14), (('i', 'did', 'not'), 14), (('i', 'saw', 'the'), 13), (('i', 'began', 'to'), 13)]

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
# Note 1: unigrams, bigrams, trigrams follow Zip's law, the Zipf law exponent decreases as the number of words increase.
# Note 2: Many n-grams occur very rarely, which makes Laplace smoothing unsuitable for language modeling. We will use
#  deep learning models for language modeling instead.


# Reading long sequence data


# 1. Random sampling
def seq_data_iter_random(corpus, batch_size, num_steps):   #@save
    """
    Generate a minibatch of subsequences using random sampling
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    """
    # Start with random offset to partition a sequence (offset in [1, num_steps-1])
    # Subtract 1, because we need to account for labels
    corpus = corpus[random.randint(0, num_steps-1):]
    num_subseqs = (len(corpus)-1) // num_steps
    # Starting indices for subsequences of length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random minibatches during iteration are not
    # necessarily adjacent on the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of lenth num_steps starting from pos
        return corpus[pos: pos+num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size*num_batches, batch_size):
        # Here, initial_indices contains randomized starting indices for subsequences
        initial_indices_per_batch = initial_indices[i: i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# Generate a sequence from 0 to 34
my_seq = list(range(34))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# X:  tensor([[ 8,  9, 10, 11, 12],
#             [28, 29, 30, 31, 32]])
# Y: tensor([[ 9, 10, 11, 12, 13],
#            [29, 30, 31, 32, 33]])
# X:  tensor([[13, 14, 15, 16, 17],
#             [23, 24, 25, 26, 27]])
# Y: tensor([[14, 15, 16, 17, 18],
#            [24, 25, 26, 27, 28]])
# X:  tensor([[18, 19, 20, 21, 22],
#             [ 3,  4,  5,  6,  7]])
# Y: tensor([[19, 20, 21, 22, 23],
#            [ 4,  5,  6,  7,  8]])


# 2. Sequential partitioning
def seq_data_iter_sequential(corpus, batch_size, num_steps): #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# X:  tensor([[ 4,  5,  6,  7,  8],
#             [18, 19, 20, 21, 22]])
# Y: tensor([[ 5,  6,  7,  8,  9],
#            [19, 20, 21, 22, 23]])
# X:  tensor([[ 9, 10, 11, 12, 13],
#             [23, 24, 25, 26, 27]])
# Y: tensor([[10, 11, 12, 13, 14],
#            [24, 25, 26, 27, 28]])


# Sequential loader class which allows using one of random sampling or sequential partitioning
class SeqDataLoader: #@save
    """
    An iterator to load sequence data.
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# Write load_data_time_machine which returns both the data iterator (using SeqLoader class) and the vocabulary
# To be used in later sections
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):  #@save
    """
    Return the iterator and the vocabulary of the time machine dataset.
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


"""
Recurrent Neural Networks
"""

# Verify that X_t * W_xh + H_(t-1) * W_hh is equal to matrix multiplication of concatenation of X_t and H_(t-1) and
# concatenation of W_xh and W_hh
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# tensor([[-2.6899, -0.0968,  1.0692,  1.6188],
#         [ 2.0035, -3.1598,  0.5432,  1.3159],
#         [-1.5754, -3.1873,  1.2767,  1.1468]])

# Concatenate X and H along columns (axis 1), W_xh and W_hh along rows (axis 0)
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
# tensor([[-2.6899, -0.0968,  1.0692,  1.6188],
#         [ 2.0035, -3.1598,  0.5432,  1.3159],
#         [-1.5754, -3.1873,  1.2767,  1.1468]])


"""
Implementation of RNNs from scratch
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
print(train_iter)
# <d2l.torch.SeqDataLoader object at 0x1140d8160>
print(vocab)
# <d2l.torch.Vocab object at 0x10802ed60>
print(len(vocab))
# 28
# Because we are using character sequences

# Each token is represented as a numerical index in train_iter
# Better representationg is one-hot encoding
# One-hot encoding vectors with indices 0 and 2 for this vocabulary is shown below
F.one_hot(torch.tensor([0, 2]), len(vocab))
# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0]])


X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)
# torch.Size([5, 2, 28])

# Initialize the model parameters


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# RNN model


def init_rnn_state(batch_size, num_hiddens, device):
    """
    init_rnn_state function to return the hidden state at initialization
    :param batch_size:
    :param num_hiddens:
    :param device:
    :return:
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    """
    rnn function defines how to compute the hidden state and output at a time step.
    Note that the RNN model loops through the outermost dimension of inputs so that it updates hidden states H
    of a minibatch, time step by time step.
    :param inputs:
    :param state:
    :param params:
    :return:
    """
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
    Y = torch.mm(H, W_hq) + b_q
    outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch: #@save
    """
    A RNN Model implemented from scratch.
    """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# Use RNN model from scratch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
# (torch.Size([2, 28]), 1, torch.Size([2, 512]))


# Prediction


def predict_ch8(prefix, num_preds, net, vocab, device): #@save
    """
    Generate new characters following the `prefix`.
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # Warm-up period
    for y in prefix[1:]:
        _, state = net(get_input(), state)
    outputs.append(vocab[y])
    # Predict `num_preds` steps
    for _ in range(num_preds):
        y, state = net(get_input(), state)
    outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# Predict following characters without training the RNN model
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
# 't h'


# Gradient clipping


def grad_clipping(net, theta): #@save
    """
    Clip the gradient.
    """
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# Training

# For one epoch only
# @save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    Train a net within one epoch (defined in Chapter 8).
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, torch.nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# For all dataset / epochs
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """
    Train a model (defined in Chapter 8).
    """
    loss = torch.nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, torch.nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


# Train RNN model
num_epochs, lr = 500, 1
# TODO
#  Fix the error "ValueError: Expected input batch_size (32) to match target batch_size (1120)."
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

# Use random sampling instead
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# TODO:
#  Fix the error "ValueError: Expected input batch_size (32) to match target batch_size (1120)."
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
#          use_random_iter=True)

"""
Concise Implementation of RNNs
"""
import torch
from torch.nn import functional as F
from d2l import torch as d2l
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = torch.nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)
# torch.Size([1, 32, 256])

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)
# torch.Size([35, 32, 256]) torch.Size([1, 32, 256])


#@save
class RNNModel(torch.nn.Module):
    """
    The RNN model.
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = torch.nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = torch.nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, torch.nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
# Out[3]: 'time travellerfq<unk>q<unk>q<unk>q<unk>q'

# Train with high-level APIs
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.4, 81965.9 tokens/sec on cpu
# time travellerit s against reason said the thimen ware mas alish
# travelleryom camp gonnt of therid and the thing to expectou

# Compared with the last section, this model achieves comparable perplexity, albeit within a shorter
# period of time, due to the code being more optimized by high-level APIs of the deep learning
# framework.

# Delete generated data folder
delete_data_folder()
