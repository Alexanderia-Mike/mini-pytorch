from typing import List, Tuple
import numpy as np

from model.tensor import Tensor
from .optim_interface import OptimInterface

class Adam(OptimInterface):
    epsilon = 1e-08

    def __init__(self, parameters:List[Tensor], learning_rate:float, 
                 betas:Tuple[float]=(0.9,0.999)) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.betas = betas
        self.m = []
        self.v = []
        self.t = 0
        for parameter in self.parameters:
            self.m.append( np.zeros(shape=parameter.shape()) )
            self.v.append( np.zeros(shape=parameter.shape()) )

    def step(self) -> None:
        self.t += 1
        for idx, parameter in enumerate(self.parameters):
            self.m[idx] = self.betas[0] * self.m[idx] + (1-self.betas[0]) * parameter.grad
            self.v[idx] = self.betas[1] * self.v[idx] + (1-self.betas[1]) * parameter.grad.__pow__(2)
            m = self.m[idx] / (1-self.betas[0] ** self.t)
            v = self.v[idx] / (1-self.betas[1] ** self.t)
            self.parameters[idx].value -= self.learning_rate * m / (v ** 0.5 + Adam.epsilon)


if __name__ == "__main__":
    import torch
    from torch.optim import Adam as nnAdam
    import numpy as np
    epsilon = 1e-6

    def prepareTest() -> Tuple[List[Tensor], float]:
        learning_rate = 0.1
        betas = (0.9, 0.999)
        np_parameters:List[Tensor] = [
            np.array([1,2,3,4], dtype=np.float64),
            np.array([[5,6],[7,8],[9,10]], dtype=np.float64),
            np.array([11,12,13], dtype=np.float64)
        ]
        np_grad = [np.random.random(p.shape) for p in np_parameters]
        my_parameters:List[Tensor] = [Tensor(np_p.copy()) for np_p in np_parameters]
        for idx, parameter in enumerate(my_parameters):
            parameter.grad = np_grad[idx]
        torch_parameters:List[torch.tensor] = [torch.tensor(np_p) for np_p in np_parameters]
        for idx, parameter in enumerate(torch_parameters):
            parameter.grad = torch.tensor(np_grad[idx])
        return my_parameters, torch_parameters, learning_rate, betas

    def adamTest():
        epochs = 10
        my_parameters, torch_parameters, lr, betas = prepareTest()
        tadam = nnAdam(params=torch_parameters, lr=lr, betas=betas)
        madam = Adam(parameters=my_parameters, learning_rate=lr, betas=betas)
        for _ in range(epochs):
            tadam.step()
            madam.step()
            for mp, tp in zip(my_parameters, torch_parameters):
                assert (mp.value - tp.numpy() < epsilon).all()
        print(f"====adamTest:====\nPassed!!!")

    def zerogradTest():
        my_parameters, _, lr, betas = prepareTest()
        madam = Adam(parameters=my_parameters, learning_rate=lr, betas=betas)
        madam.step()
        madam.zero_grad()
        for mp in my_parameters:
            if mp.requires_grad:
                assert (mp.grad.value == 0).all()
        print(f"====adamzerogradTest:====\nPassed!!!")

    
    adamTest()
    zerogradTest()