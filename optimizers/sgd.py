from typing import List, Tuple
import numpy as np

from model.tensor import Tensor
from .optim_interface import OptimInterface

class SGD(OptimInterface):
    def __init__(self, parameters:List[Tensor], learning_rate:float, friction:float=0.9) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.friction = friction
        self.velocities = []
        for parameter in self.parameters:
            self.velocities.append( np.zeros(shape=parameter.shape()) )

    def step(self) -> None:
        for idx, parameter in enumerate(self.parameters):
            self.velocities[idx] = self.friction * self.velocities[idx] + parameter.grad
            self.parameters[idx].value -= self.velocities[idx] * self.learning_rate


if __name__ == "__main__":
    import torch
    from torch.optim import SGD as nnSGD
    import numpy as np

    def prepareTest() -> Tuple[List[Tensor], float]:
        learning_rate = 0.1
        momentum = 0.9
        np_parameters:List[Tensor] = [
            np.array([1,2,3,4], dtype=np.float64),
            np.array([[5,6],[7,8],[9,10]], dtype=np.float64),
            np.array([11,12,13], dtype=np.float64)
        ]
        np_grad = [np.random.random(p.shape) for p in np_parameters]
        my_parameters:List[Tensor] = [Tensor(np_p) for np_p in np_parameters]
        for idx, parameter in enumerate(my_parameters):
            parameter.grad = np_grad[idx]
        torch_parameters:List[torch.tensor] = [torch.tensor(np_p) for np_p in np_parameters]
        for idx, parameter in enumerate(torch_parameters):
            parameter.grad = torch.tensor(np_grad[idx])
        return my_parameters, torch_parameters, learning_rate, momentum

    def gdnormalTest():
        my_parameters, torch_parameters, lr, _ = prepareTest()
        tgd = nnSGD(params=torch_parameters, lr=lr)
        mgd = SGD(parameters=my_parameters, learning_rate=lr)
        tgd.step()
        mgd.step()
        for mp, tp in zip(my_parameters, torch_parameters):
            assert (mp.value == tp.numpy()).all()
        print(f"====gdnormalTest:====\nPassed!!!")

    def gdmomentumTest():
        epochs = 10
        my_parameters, torch_parameters, lr, momentum = prepareTest()
        tgdm = nnSGD(params=torch_parameters, lr=lr, momentum=momentum)
        mgdm = SGD(parameters=my_parameters, learning_rate=lr, friction=momentum)
        for _ in range(epochs):
            tgdm.step()
            mgdm.step()
            for mp, tp in zip(my_parameters, torch_parameters):
                assert (mp.value == tp.numpy()).all()
        print(f"====gdmomentumTest:====\nPassed!!!")

    def zerogradTest():
        my_parameters, _, lr, _ = prepareTest()
        msgd = SGD(parameters=my_parameters, learning_rate=lr)
        msgd.step()
        msgd.zero_grad()
        for mp in my_parameters:
            if mp.requires_grad:
                assert (mp.grad.value == 0).all()
        print(f"====sgdzerogradTest:====\nPassed!!!")
    
    gdnormalTest()
    gdmomentumTest()
    zerogradTest()