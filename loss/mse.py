from model.tensor import Tensor
from .loss_interface import LossInterface

class MSELoss(LossInterface):
    def __init__(self) -> None:
        pass

    def __call__(self, predicted:Tensor, target:Tensor) -> Tensor:
        target_copy = Tensor(target.value.reshape(predicted.shape()))
        n = predicted.size()
        return (predicted - target_copy).__pow__(2).sum() / n
    

if __name__ == "__main__":
    import torch
    from torch.nn import MSELoss as nnMSELoss
    import numpy as np
    epsilon = 1e-6

    def mselossTest1():
        nnLoss = nnMSELoss()
        myLoss = MSELoss()
        target = np.array([1,2,3,4], dtype=np.float64)
        predicted = np.array([1.1,2.2,2.8,4.3], dtype=np.float64)
        tt = torch.tensor(target)
        tp = torch.tensor(predicted, requires_grad=True)
        mt = Tensor(target)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnLoss(tp, tt)
        ml = myLoss(mp, mt)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl.backward()
        ml.backward()
        assert (mp.grad == tp.grad.numpy()).all()
        print(f"====MSELossTest1:====\nPassed!!!")
    
    def mselossTest2():
        nnLoss = nnMSELoss()
        myLoss = MSELoss()
        target = 10 * np.random.rand(10,3) + 50
        noise = np.random.random(target.shape)
        predicted = target + noise
        tt = torch.tensor(target)
        tp = torch.tensor(predicted, requires_grad=True)
        mt = Tensor(target)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnLoss(tp, tt)
        ml = myLoss(mp, mt)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl.backward()
        ml.backward()
        assert (mp.grad == tp.grad.numpy()).all()
        print(f"====MSELossTest2:====\nPassed!!!")

    mselossTest1()
    mselossTest2()