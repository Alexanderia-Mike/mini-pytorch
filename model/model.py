from typing import List

from .tensor import Tensor
from .nn import LayerInterface

class model:
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, x:Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        parameters = []
        for _, value in self.__dict__.items():
            if isinstance(value, LayerInterface):
                parameters += value.getParameters()
        return parameters


if __name__ == "__main__":
    # integration test for model
    from .nn import Linear, ReLU
    from initializers.utils import Initializer
    from dataset.numpyNN import sample_data
    from loader.data_loader import DataLoader
    from optimizers.adam import Adam
    from loss.mse import MSELoss
    from loss.cross_entropy import CrossEntropyLoss

    from typing import Tuple

    def getData(data_name:str) -> Tuple[Tensor, Tensor]:
        train_data, train_target, test_data, test_target = sample_data(data_name)
        return DataLoader(train_data, train_target, test_data, test_target).getData()
    
    def regressionTest():
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
        optimizer = Adam(my_model.parameters(), learning_rate=lr)
        criterion = MSELoss()

        print(f"====regressionTest:====")
        # train
        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted = my_model(train_data)
            loss = criterion(predicted, train_target)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}: loss = {loss}")
        print(f"train_loss: {loss}")

        # test
        predicted = my_model(test_data)
        loss = criterion(predicted, test_target)
        print(f"test loss: {loss}")

    def classificationTest():
        train_data, train_target, test_data, test_target = getData(data_name="circle")
        data_dim = train_data.shape()[1]
        target_dim = 2  # two categories

        class classificationModel(model):
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

        my_model = classificationModel()
        optimizer = Adam(my_model.parameters(), learning_rate=lr)
        criterion = CrossEntropyLoss()

        print(f"====classificationTest:====")
        # train
        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted = my_model(train_data)
            loss = criterion(predicted, train_target)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}: loss = {loss}")
        print(f"train_loss: {loss}")

        # test
        predicted = my_model(test_data)
        loss = criterion(predicted, test_target)
        print(f"test loss: {loss}")
    
    regressionTest()
    classificationTest()