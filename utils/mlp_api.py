from typing import Tuple

from model.model import model
from model.tensor import Tensor
from loss.utils import Loss
from loss.mse import MSELoss
from loss.cross_entropy import CrossEntropyLoss
from optimizers.utils import Optimizer
from optimizers.sgd import SGD
from optimizers.adam import Adam

def train_mlp(mlp:model, train_data_set:Tuple[Tensor], num_epochs:int, 
              lr:float, loss_method:Loss, optimizer_method:Optimizer, log=False) -> float:
    train_data, train_target = train_data_set
    criterion = MSELoss() if loss_method is Loss.MSE else CrossEntropyLoss()
    optimizer = SGD(mlp.parameters(), learning_rate=lr, friction=0) \
                    if optimizer_method is Optimizer.DG_NORMAL else \
                SGD(mlp.parameters(), learning_rate=lr) \
                    if optimizer_method is Optimizer.DG_MOMENTUM else \
                Adam(mlp.parameters(), learning_rate=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted = mlp(train_data)
        loss = criterion(predicted, train_target)
        loss.backward()
        optimizer.step()
        if log: print(f"epoch {epoch}: loss = {loss.value.item(): .2f}")
    if log: print(f"train_loss: {loss.value.item(): .2f}")
    return loss.value.item()

def test_mlp(mlp:model, test_data_set:Tuple[Tensor], loss_method:Loss, 
             regression:bool=True, log:bool=False) -> Tuple[float, float]:
    test_data, test_target = test_data_set
    criterion = MSELoss() if loss_method is Loss.MSE else CrossEntropyLoss()
    data_count = test_target.shape()[0]
    predicted = mlp(test_data)
    loss = criterion(predicted, test_target)
    if regression:
        correct = ((predicted.value > 0.5) & (test_target.value.reshape(predicted.shape()) == 1) | \
                  (predicted.value < 0.5) & (test_target.value.reshape(predicted.shape()) == 0)).sum().item()
    else:
        prediction = predicted.value.argmax(axis=-1)
        correct = (prediction == test_target.value).sum().item()
    acc = correct / data_count
    if log: print(f"test loss: {loss.value.item(): .2f}\naccuracy: {100 * acc: .2f}%")
    return loss.value.item(), acc


if __name__ == "__main__":
    from model.nn import Linear, ReLU
    from initializers.utils import Initializer
    from dataset.numpyNN import sample_data
    from loader.data_loader import DataLoader
    from optimizers.adam import Adam
    from optimizers.utils import Optimizer
    from loss.mse import MSELoss
    from loss.utils import Loss

    from typing import Tuple

    def getData(data_name:str) -> Tuple[Tensor, Tensor]:
        train_data, train_target, test_data, test_target = sample_data(data_name)
        return DataLoader(train_data, train_target, test_data, test_target).getData()

    train_data, train_target, test_data, test_target = getData(data_name="circle")
    data_dim = train_data.shape()[1]
    target_dim = 1

    class regressionModel(model):
        def __init__(self) -> None:
            self.fc1 = Linear(in_dim=data_dim, out_dim=10, init_method=Initializer.HE)
            self.act1 = ReLU()
            self.fc2 = Linear(in_dim=10, out_dim=20, init_method=Initializer.HE)
            self.act2 = ReLU()
            self.fc3 = Linear(in_dim=20, out_dim=target_dim, init_method=Initializer.HE)
        
        def forward(self, x: Tensor) -> Tensor:
            output = self.fc1(x)
            output = self.act1(output)
            output = self.fc2(output)
            output = self.act2(output)
            output = self.fc3(output)
            return output
    
    epochs = 20
    lr = 0.01
    my_model = regressionModel()

    print(f"====mlpapiTest:====")
    train_mlp(mlp=my_model, 
              train_data_set=(train_data, train_target), 
              num_epochs=epochs, 
              lr=lr, 
              loss_method=Loss.MSE, 
              optimizer_method=Optimizer.ADAM,
              log=True)

    test_mlp(mlp=my_model,
             test_data_set=(test_data, test_target),
             loss_method=Loss.MSE,
             log=True)