import numpy as np
from typing import List
from .tensor import Tensor

from initializers.he_initializer import HeInitializer
from initializers.xavier_initializer import XavierInitializer
from initializers.utils import Initializer as INIT_METHOD
from model.tensor import Tensor
from model.predecessor import Predecessor

class LayerInterface:
    def __init__(self) -> None:
        pass

    def getParameters(self) -> List[Tensor]:
        raise NotImplementedError
    
    def __call__(self, x:Tensor) -> Tensor:
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return self.__str__()

class Linear(LayerInterface):
    def __init__(self, in_dim:int, out_dim:int, init_method:INIT_METHOD) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        initializer = HeInitializer(in_dim) if init_method == INIT_METHOD.HE else \
                      XavierInitializer(in_dim)
        weights = np.zeros(shape=[out_dim, in_dim])
        bias = np.zeros(out_dim)
        for row in weights:
            for cid, _ in enumerate(row):
                row[cid] = initializer()
        for cid, _ in enumerate(bias):
            bias[cid] = initializer()
        self.weights = Tensor(weights, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

    def getParameters(self) -> List[Tensor]:
        return [self.weights, self.bias]

    def __call__(self, x:Tensor) -> Tensor:
        return x @ self.weights.transpose(0,1) + self.bias
    
    def __str__(self) -> str:
        return f"weights:\n{self.weights}\nbias:\n{self.bias}"


class ReLU:
    def __init__(self) -> None:
        pass

    def __call__(self, x:Tensor) -> Tensor:
        kept = x.value > 0
        result = Tensor(np.where(kept, x.value, 0), requires_grad=x.requires_grad)
        result._appendPredecessor(x, lambda x: np.where(kept, x, 0))
        return result


class Sigmoid:
    def __init__(self) -> None:
        pass

    def __call__(self, x:Tensor) -> Tensor:
        return 1 / (1 + Tensor.exp(-x))
    

class Tanh:
    def __init__(self) -> None:
        pass

    def __call__(self, x:Tensor) -> Tensor:
        return (Tensor.exp(x) - Tensor.exp(-x)) / (Tensor.exp(x) + Tensor.exp(-x))
    

class NoActivation:
    def __init__(self) -> None:
        pass

    def __call__(self, x:Tensor) -> Tensor:
        return x
    


if __name__ == "__main__":
    import torch
    from torch.nn import ReLU as nnReLU, Sigmoid as nnSigmoid, Tanh as nnTanh
    epsilon = 1e-6

    def linearTest1():
        l = Linear(3, 4, INIT_METHOD.HE)
        x = Tensor(np.array([1,2,3]), requires_grad=True)
        r = l(x)
        print(f"====LinearTest1:====\nresult is {r}")
    
    def linearTest2():
        l = Linear(3, 4, INIT_METHOD.HE)
        x = Tensor(np.array(list(range(30))).reshape(10,3), requires_grad=True)
        r = l(x)
        print(f"====LinearTest2:====\ndata is\n{x}\nresult is\n{r}")

    def reluTest1():
        tr = nnReLU()
        tx = torch.tensor([-1,-2,-3,1,2,3], dtype=torch.float32, requires_grad=True)
        ty = tr(tx).sum()
        mr = ReLU()
        mx = Tensor(np.array([-1,-2,-3,1,2,3]), requires_grad=True)
        my = mr(mx).sum()
        assert (my.value == ty.detach().numpy()).all()
        ty.backward()
        my.backward()
        assert (mx.grad == tx.grad.numpy()).all()
        print(f"====ReluTest1:====\nPassed!!!")
    
    def reluTest2():
        x = np.random.rand(10,3)
        tr = nnReLU()
        tx = torch.tensor(x, requires_grad=True)
        ty = tr(tx).sum()
        mr = ReLU()
        mx = Tensor(x, requires_grad=True)
        my = mr(mx).sum()
        assert (my.value - ty.detach().numpy() < epsilon).all()
        ty.backward()
        my.backward()
        assert (mx.grad - tx.grad.numpy() < epsilon).all()
        print(f"====ReluTest2:====\nPassed!!!")

    def sigmoidTest():
        x = np.array([-1,-2,-3,1,2,3], dtype=np.float64)
        ts = nnSigmoid()
        tx = torch.tensor(x, requires_grad=True)
        ty = ts(tx).sum()
        ms = Sigmoid()
        mx = Tensor(x, requires_grad=True)
        my = ms(mx).sum()
        assert (my.value - ty.detach().numpy() < epsilon).all()
        ty.backward()
        my.backward()
        assert (mx.grad - tx.grad.numpy() < epsilon).all()
        print(f"====SigmoidTest:====\nPassed!!!")
    
    def tanhTest():
        x = np.array([-1,-2,-3,1,2,3], dtype=np.float64)
        ts = nnTanh()
        tx = torch.tensor(x, requires_grad=True)
        ty = ts(tx).sum()
        ms = Tanh()
        mx = Tensor(x, requires_grad=True)
        my = ms(mx).sum()
        assert (my.value - ty.detach().numpy() < epsilon).all()
        ty.backward()
        my.backward()
        assert (mx.grad - tx.grad.numpy() < epsilon).all()
        print(f"====TanhTest:====\nPassed!!!")

    def noactivationTest():
        x = np.array([-1,-2,-3,1,2], dtype=np.float64)
        ts = lambda x: x
        tx = torch.tensor(x, requires_grad=True)
        ty = ts(tx).sum()
        ms = NoActivation()
        mx = Tensor(x, requires_grad=True)
        my = ms(mx).sum()
        assert (my.value - ty.detach().numpy() < epsilon).all()
        ty.backward()
        my.backward()
        assert (mx.grad - tx.grad.numpy() < epsilon).all()
        print(f"====NoActivationTest:====\nPassed!!!")

    linearTest1()
    linearTest2()
    reluTest1()
    reluTest2()
    sigmoidTest()
    tanhTest()
    noactivationTest()
