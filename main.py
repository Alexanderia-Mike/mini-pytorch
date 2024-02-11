from model.nn import Linear, ReLU
from model.tensor import Tensor
from model.model import model
from initializers.utils import Initializer
from dataset.numpyNN import sample_data
from loader.data_loader import DataLoader
from optimizers.utils import Optimizer
from loss.utils import Loss
from utils.mlp_api import train_mlp, test_mlp
from utils.plot import plot_loss, plot_decision_boundary

from typing import Tuple

def getData(data_name:str) -> Tuple[Tensor, Tensor]:
    train_data, train_target, test_data, test_target = sample_data(data_name)
    return DataLoader(train_data, train_target, test_data, test_target).getData()

dataset = getData(data_name="XOR")
data_dim = dataset[0].shape()[1]
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

epochs = 500
lr = 0.001
loss_method = Loss.MSE
optimizer_method = Optimizer.ADAM
my_model = regressionModel()

train_data, train_target, test_data, test_target = dataset
train_loss = []
test_loss = []
for _ in range(epochs):
    train_loss.append(
        train_mlp(mlp=my_model, 
                    train_data_set=(train_data, train_target), 
                    num_epochs=1,lr=lr, 
                    loss_method=loss_method, 
                    optimizer_method=optimizer_method)
    )

    tl, _ = test_mlp(mlp=my_model,
                        test_data_set=(test_data, test_target),
                        loss_method=loss_method,
                        regression=True)
    test_loss.append(tl)

plot_loss(
    filename=f"./loss",
    logs={
        "epochs": list(range(epochs)),
        "train_loss": train_loss,
        "test_loss": test_loss
})
plot_decision_boundary(
    filename=f"./boundary",
    X=train_data.value,
    y=train_target.value,
    pred_fn=lambda x: my_model(Tensor(x)).value,
    boundry_level=20
)