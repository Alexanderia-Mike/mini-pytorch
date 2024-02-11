import numpy as np
from typing import Tuple

from model.tensor import Tensor

class DataLoader:
    def __init__(self, train_data:np.ndarray, train_target:np.ndarray,
                 test_data:np.ndarray, test_target:np.ndarray) -> None:
        self.train_data = Tensor(train_data, requires_grad=False)
        self.train_target = Tensor(train_target.reshape(-1), requires_grad=False)
        self.test_data = Tensor(test_data, requires_grad=False)
        self.test_target = Tensor(test_target.reshape(-1), requires_grad=False)
    
    def getData(self) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        return self.train_data, self.train_target, self.test_data, self.test_target