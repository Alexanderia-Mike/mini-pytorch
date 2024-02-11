# mini-pytorch

Author: Chenfei Lou (chenfeil@andrew.cmu.edu)

## overview
this is a toy implementation of Python PyTorch library. It implements a very limited subsets of functionalities of PyTorch, nevertheless it embodies all the basic features of PyTorch such as tensor, model, loss functions and optimizers, and it keeps all the APIs the same as in PyTorch. Most importantly, it implements the automatic gradient backward propagations, which is the essential soul of PyTorch library. With simple and easily-readable codes, this repository aims to serve as an entry point for those who hopes to get a taste of how PyTorch achieves its magic gradient propagation, and how its high-level ML APIs are implemented.

## Content
this repo includes the following parts:
- datasets: example data sets for training and testing
    - "XOR"
    - "linear-separable"
    - "circle"
    - "sinusoid"
    - "swiss-roll"
- initializers: how parameters in ML layers are initialized
    - He initialization
    - Xavier initialization
- loader: provide APIs to load datasets to tensors
- loss: loss functions
    - cross entropy loss
    - mean squared error loss
- model:
    - model: an interface for neural network models
    - nn: neural network layers and activation functions
        - linear
        - ReLU
        - Sigmoid
        - TanH
    - tensor: tensor class that can optionally enable automatic gradient propagations.
- optimizers:
    - sgd: stochastic gradient descent
    - adam: Adam optimizer
- utils: some APIs for convenience
    - mlp_api: for training and testing
    - plot: for plotting loss diagram and decision boundary countour diagrams.

## Example Usage
you can use the modules in this repo to build and train a neural network pretty much the same way as you would using PyTorch library. The following codes show how you could build a 3-layer neural network, then train and test it using XOR dataset:

```python
from model.nn import Linear, ReLU
from model.tensor import Tensor
from model.model import model
from initializers.utils import Initializer
from dataset.numpyNN import sample_data
from loader.data_loader import DataLoader
from optimizers.adam import Adam
from loss.mse import MSELoss

from typing import Tuple

def getData(data_name:str) -> Tuple[Tensor, Tensor]:
    train_data, train_target, test_data, test_target = sample_data(data_name)
    return DataLoader(train_data, train_target, test_data, test_target).getData()

train_data, train_target, test_data, test_target = getData(data_name="XOR")
data_dim = train_data.shape()[-1]
target_dim = 1

class regressionModel(model):
    def __init__(self) -> None:
        self.fc1 = Linear(in_dim=data_dim, out_dim=10, init_method=Initializer.HE)
        self.act1 = ReLU()
        self.fc2 = Linear(in_dim=10, out_dim=target_dim, init_method=Initializer.HE)

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc1(x)
        output = self.act1(output)
        output = self.fc2(output)
        return output

epochs = 1000
lr = 0.001
my_model = regressionModel()
criterion = MSELoss()
optimizer = Adam(my_model.parameters(), learning_rate=lr)

# train
for epoch in range(epochs):
    optimizer.zero_grad()
    predicted = my_model(train_data)
    loss = criterion(predicted, train_target)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss = {loss.value.item(): .2f}")
print(f"train_loss: {loss.value.item(): .2f}")

# test
data_count = test_target.shape()[0]
predicted = my_model(test_data)
loss = criterion(predicted, test_target)
correct = ((predicted.value > 0.5) & (test_target.value.reshape(predicted.shape()) == 1) | \
           (predicted.value < 0.5) & (test_target.value.reshape(predicted.shape()) == 0)).sum().item()
acc = correct / data_count
print(f"test loss: {loss.value.item(): .2f}\naccuracy: {100 * acc: .2f}%")
```

You can find more examples about how to use APIs to train, test and plot diagrams in "main.py".