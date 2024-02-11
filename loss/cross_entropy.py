import numpy as np
from model.tensor import Tensor
from .loss_interface import LossInterface

class CrossEntropyLoss(LossInterface):
    def __init__(self) -> None:
        pass

    def _softMax(predicted:Tensor) -> Tensor:
        shape_len = len(predicted.shape())
        exp = predicted.exp()
        return exp / exp.sum(dim=shape_len-1) \
                        .expand_axis(dims=shape_len-1)

    def __call__(self, predicted:Tensor, target:Tensor, mean=True) -> Tensor:
        # transform target to one-hot vector
        class_num = predicted.shape()[-1]
        one_hot_target = np.zeros(shape=(*target.shape(),class_num))
        if len(one_hot_target.shape) > 1:
            one_hot_target[np.arange(target.size()), target.value.astype(np.int64)] = 1
        else:
            one_hot_target[target.value] = 1
        one_hot_target = Tensor(one_hot_target)
        # calculate loss
        shape_len = len(predicted.shape())
        batch_size = 1 if shape_len == 1 else predicted.shape()[0]
        soft_max = CrossEntropyLoss._softMax(predicted)
        batch_ce = one_hot_target * soft_max.log()
        if mean:
            return -batch_ce.sum() / batch_size
        return -batch_ce.sum()


if __name__ == "__main__":
    import torch
    from torch.nn import CrossEntropyLoss as nnCrossEntropyLoss, Softmax as nnSoftmax
    import numpy as np
    epsilon = 1e-6

    def softmaxTest1():
        nnsm = nnSoftmax(0)
        mysm = CrossEntropyLoss._softMax
        predicted = np.array([0.9,0.2,0.1,0.8], dtype=np.float64)
        tp = torch.tensor(predicted, requires_grad=True)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnsm(tp)
        ml = mysm(mp)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl = tl.sum()
        ml = ml.sum()
        tl.backward()
        ml.backward()
        assert (mp.grad - tp.grad.numpy() < epsilon).all()
        print(f"====SoftmaxTest1:====\nPassed!!!")
    
    def softmaxTest2():
        nnsm = nnSoftmax(1)
        mysm = CrossEntropyLoss._softMax
        predicted = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
        tp = torch.tensor(predicted, requires_grad=True)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnsm(tp)
        ml = mysm(mp)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl = tl.sum()
        ml = ml.sum()
        tl.backward()
        ml.backward()
        assert (mp.grad - tp.grad.numpy() < epsilon).all()
        print(f"====SoftmaxTest2:====\nPassed!!!")

    def crossentropylossTest1():
        nnLoss = nnCrossEntropyLoss()
        myLoss = CrossEntropyLoss()
        target = np.array(0, dtype=np.int64)
        predicted = np.array([0.9,0.2,0.1,0.8], dtype=np.float64)
        tt = torch.tensor(target)
        tp = torch.tensor(predicted, requires_grad=True)
        mt = Tensor(target)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnLoss(tp, tt)
        ml = myLoss(mp, mt)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl.backward()
        ml.backward()
        assert (mp.grad - tp.grad.numpy() < epsilon).all()
        print(f"====CrossEntropyLossTest1:====\nPassed!!!")
    
    def crossentropylossTest2():
        nnLoss = nnCrossEntropyLoss()
        myLoss = CrossEntropyLoss()
        target = np.array([0,2,1], dtype=np.int64)
        predicted = np.random.rand(3,4)
        tt = torch.tensor(target)
        tp = torch.tensor(predicted, requires_grad=True)
        mt = Tensor(target)
        mp = Tensor(predicted, requires_grad=True)
        tl = nnLoss(tp, tt)
        ml = myLoss(mp, mt)
        assert (ml.value - tl.detach().numpy() < epsilon).all()
        tl.backward()
        ml.backward()
        assert (mp.grad - tp.grad.numpy() < epsilon).all()
        print(f"====CrossEntropyLossTest2:====\nPassed!!!")

    softmaxTest1()
    softmaxTest2()
    crossentropylossTest1()
    crossentropylossTest2()