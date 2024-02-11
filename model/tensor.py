import numpy as np
from typing import List, Union, Tuple, Callable

from .predecessor import Predecessor

class Tensor:
    def _broadcastable(a:np.ndarray, b:np.ndarray) -> [bool, Tuple,Tuple]:
        """
        returns the dimensions corresponding to a and b, respectively, that 
        should be summed after the gradient is derived
        """
        a_shape, b_shape = a.shape, b.shape
        a_axis, b_axis = [], []
        max_len = max(len(a_shape), len(b_shape))
        for i in range(max_len):
            if i >= len(a_shape):
                a_axis.append(max_len - i - 1)
            elif i >= len(b_shape):
                b_axis.append(max_len - i - 1)
            elif a_shape[~i] != b_shape[~i]:
                if a_shape[~i] == 1:    a_axis.append(max_len - i - 1)
                elif b_shape[~i] == 1:  b_axis.append(max_len - i - 1)
                else:
                    return False, None, None
        return True, tuple(a_axis), tuple(b_axis)

    def _scalarToTensor(other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array([other]), requires_grad=False)
        if not isinstance(other, Tensor):
            raise Exception(f"unsupported operation between Tensor and {type(other)}")
        return other

    def __init__(self, value:np.ndarray, requires_grad=False) -> None:
        self.value = value
        self.requires_grad = requires_grad
        self.predecessors:List[Predecessor] = []
        self.grad = np.zeros(self.value.shape) if requires_grad else None
        self.suc_count = 0

    def _appendPredecessor(self, tensor:"Tensor", func:Callable) -> None:
        if tensor.requires_grad:
            predecessor = Predecessor(tensor, func)
            self.predecessors.append(predecessor)
            predecessor.predecessor.suc_count += 1

    def shape(self):
        return self.value.shape
    
    def size(self):
        return self.value.size

    def __add__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        broadcastable, a_axis, b_axis = Tensor._broadcastable(self.value, other.value)
        if not broadcastable:
            raise Exception("dimension incompatible!")
        result = Tensor(self.value + other.value, requires_grad = self.requires_grad or other.requires_grad)
        result._appendPredecessor(
            self, lambda x: x.sum(axis=a_axis).reshape(self.value.shape))
        result._appendPredecessor(
            other, lambda x: x.sum(axis=b_axis).reshape(other.value.shape))
        return result
    
    def __radd__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        return other.__add__(self)

    def __neg__(self):
        result = Tensor(-self.value, requires_grad = self.requires_grad)
        result._appendPredecessor(self, lambda x: -x)
        return result

    def __sub__(self, other:Union["Tensor",int,float]) -> "Tensor":
        return self.__add__(other.__neg__())
    
    def __rsub__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        return other.__sub__(self)

    def __mul__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        broadcastable, a_axis, b_axis = Tensor._broadcastable(self.value, other.value)
        if not broadcastable:
            raise Exception("dimension incompatible!")
        result = Tensor(self.value * other.value, requires_grad = self.requires_grad or other.requires_grad)
        result._appendPredecessor(
            self, lambda x: (x * other.value).sum(axis=a_axis).reshape(self.value.shape))
        result._appendPredecessor(
            other, lambda x: (x * self.value).sum(axis=b_axis).reshape(other.value.shape))
        return result

    def __rmul__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        return other.__mul__(self)

    def __truediv__(self, other:"Tensor") -> "Tensor":
        other = Tensor._scalarToTensor(other)
        broadcastable, a_axis, b_axis = Tensor._broadcastable(self.value, other.value)
        if not broadcastable:
            raise Exception("dimension incompatible!")
        result = Tensor(self.value / other.value, requires_grad = self.requires_grad or other.requires_grad)
        result._appendPredecessor(
            self, lambda x: (x / other.value).sum(axis=a_axis).reshape(self.value.shape))
        result._appendPredecessor(
            other, lambda x: (-x * self.value / other.value.__pow__(2))
                                                           .sum(axis=b_axis)
                                                           .reshape(other.value.shape))
        return result

    def __rtruediv__(self, other:Union["Tensor",int,float]) -> "Tensor":
        other = Tensor._scalarToTensor(other)
        return other.__truediv__(self)
    
    def __pow__(self, power:int|float) -> "Tensor":
        result = Tensor(self.value.__pow__(power), requires_grad = self.requires_grad)
        result._appendPredecessor(
            self, lambda x: x * power * self.value.__pow__(power - 1))
        return result

    def sum(self, dim=None) -> "Tensor":
        if dim is None:
            result = Tensor(self.value.sum(), requires_grad = self.requires_grad)
            result._appendPredecessor(self, lambda x: x * np.ones(self.value.shape))
            return result
        result = Tensor(self.value.sum(axis=dim), requires_grad = self.requires_grad)
        result._appendPredecessor(
            self, lambda x: np.broadcast_to(array=np.expand_dims(x, axis=dim), 
                                            shape=self.value.shape))
        return result
    
    def __matmul__(self, other:"Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise Exception(f"unsupported operation between Tensor and {type(other)}")
        if self.value.shape[-1] != other.value.shape[0]:
            raise Exception("dimension incompatible!")
        result = Tensor(self.value @ other.value, requires_grad = self.requires_grad or other.requires_grad)
        if len(self.value.shape) == 1 and len(other.value.shape) == 1:
            result._appendPredecessor(self, lambda x: x * other.value)
            result._appendPredecessor(other, lambda x: x * self.value)
        elif len(self.value.shape) == 1:
            result._appendPredecessor(self, lambda x: x @ other.value.transpose())
            result._appendPredecessor(other, lambda x: np.outer(self.value, x))
        elif len(other.value.shape) == 1:
            result._appendPredecessor(self, lambda x: np.outer(x, other.value))
            result._appendPredecessor(other, lambda x: self.value.transpose() @ x)
        else:
            result._appendPredecessor(self, lambda x: x @ other.value.transpose())
            result._appendPredecessor(other, lambda x: self.value.transpose() @ x)
        return result
    
    def transpose(self, dim1:int, dim2:int) -> "Tensor":
        permutations = list(range(len(self.value.shape)))
        permutations[dim1] = dim2
        permutations[dim2] = dim1
        result = Tensor(self.value.transpose(permutations), requires_grad = self.requires_grad)
        result._appendPredecessor(self, lambda x: x.transpose(permutations))
        return result
    
    def exp(tensor:"Tensor") -> "Tensor":
        result = Tensor(np.exp(tensor.value), requires_grad = tensor.requires_grad)
        result._appendPredecessor(tensor, lambda x: x * np.exp(tensor.value))
        return result
    
    def log(tensor:"Tensor") -> "Tensor":
        result = Tensor(np.log(tensor.value), requires_grad = tensor.requires_grad)
        result._appendPredecessor(tensor, lambda x: x * 1 / tensor.value)
        return result
    
    def expand_axis(self, dims:int|Tuple[int]):
        result = Tensor(np.expand_dims(self.value, dims), requires_grad = self.requires_grad)
        result._appendPredecessor(self, lambda x: np.squeeze(x, dims))
        return result

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros(self.value.shape)
    
    def backward(self, initial=True) -> None:
        if not initial:
            for p in self.predecessors:
                p.predecessor.grad += p.transformer(self.grad)
            return
        self.grad = np.array(1)
        indegrees = {}
        sources = [self]
        while sources:
            next_source = sources.pop()
            next_source.backward(initial=False)
            for p in next_source.predecessors:
                predecessor = p.predecessor
                if predecessor not in indegrees.keys():
                    indegrees[predecessor] = predecessor.suc_count - 1
                else:
                    indegrees[predecessor] -= 1
                if indegrees[predecessor] == 0:
                    sources.append(predecessor)

    def __str__(self) -> str:
        return self.value.__str__()
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self) -> int:
        return self.value.__len__()


# tests
if __name__ == "__main__":
    import torch
    epsilon = 1e-6

    def arrayEqual(a1:np.ndarray, a2:np.ndarray) -> bool:
        return (a1 - a2 < epsilon).all()
    
    def prepareUnitTest():
        ta = torch.tensor(a, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True)
        ma = Tensor(a.copy(), requires_grad=True)
        mb = Tensor(b.copy(), requires_grad=True)
        return ta, tb, ma, mb
    
    a = np.array([1,2,3,4], dtype=np.float64)
    b = np.array([5,6,7,8], dtype=np.float64)
    
    def addTest():
        ta, tb, ma, mb = prepareUnitTest()
        tc = (ta + tb).sum()
        mc = (ma + mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())
    
    def subTest():
        ta, tb, ma, mb = prepareUnitTest()
        tc = (ta - tb).sum()
        mc = (ma - mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())

    def mulTest():
        ta, tb, ma, mb = prepareUnitTest()
        tc = (ta * tb).sum()
        mc = (ma * mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())

    def divTest():
        a = np.array([1, 2], dtype=np.float64)
        b = np.array([3], dtype=np.float64)
        ta = torch.tensor(a, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True)
        ma = Tensor(a, requires_grad=True)
        mb = Tensor(b, requires_grad=True)
        tc = (ta / tb).sum()
        mc = (ma / mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())

    def powTest():
        power = 2.5
        a = np.array([1,2,3,4])
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tc = (ta ** power).sum()
        ma = Tensor(a, requires_grad=True)
        mc = (ma ** power).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())

    def sumTest():
        x = np.random.rand(2,3,4)
        tx = torch.tensor(x, requires_grad=True)
        ty = tx.sum(dim=0)
        tc = ty.sum()
        mx = Tensor(x, requires_grad=True)
        my = mx.sum(dim=0)
        mc = my.sum()
        assert arrayEqual(my.value, ty.detach().numpy())
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(mx.grad, tx.grad.numpy())

    def transposeTest():
        a = np.array([[1,2],[3,4],[5,6]])
        b = np.array([[7,8],[9,10],[11,12]])
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        tc = (ta.transpose(0,1) @ tb).sum()
        ma = Tensor(a, requires_grad=True)
        mb = Tensor(b, requires_grad=True)
        mc = (ma.transpose(0,1) @ mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())

    def matmulTest1():
        ta, tb, ma, mb = prepareUnitTest()
        tc = ta @ tb
        mc = ma @ mb
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())
    
    def matmulTest2():
        a = np.array([1, 2], dtype=np.float64)
        b = np.array([[3, 4, 5],[6, 7, 8]], dtype=np.float64)
        ta = torch.tensor(a, requires_grad=True)
        tb = torch.tensor(b, requires_grad=True)
        ma = Tensor(a, requires_grad=True)
        mb = Tensor(b, requires_grad=True)
        tc = (ta @ tb).sum()
        mc = (ma @ mb).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())
        assert arrayEqual(mb.grad, tb.grad.numpy())

    def expTest():
        a = np.array([1,2,3,4])
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tc = torch.exp(ta).sum()
        ma = Tensor(a, requires_grad=True)
        mc = Tensor.exp(ma).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())

    def logTest():
        a = np.array([1,2,3,4])
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tc = torch.log(ta).sum()
        ma = Tensor(a, requires_grad=True)
        mc = Tensor.log(ma).sum()
        assert arrayEqual(mc.value, tc.detach().numpy())
        tc.backward()
        mc.backward()
        assert arrayEqual(ma.grad, ta.grad.numpy())

    def broadcastUnitTest():
        a = np.random.rand(2,3,4)
        b = np.random.rand(4)
        assert Tensor._broadcastable(a, b) == (True, (), (1,0))
        a = np.random.rand(2,3,4)
        b = np.random.rand(2,1,4)
        assert Tensor._broadcastable(a, b) == (True, (), (1,))
        a = np.random.rand(1,3,4)
        b = np.random.rand(2,1,4)
        assert Tensor._broadcastable(a, b) == (True, (0,), (1,))
        a = np.random.rand(1,3,4)
        b = np.random.rand(2,2,4)
        assert Tensor._broadcastable(a, b) == (False, None, None)

    def broadcastTest1():
        a = np.array([[1,2]], dtype=np.float64)
        b = np.array([[3],[4]], dtype=np.float64)
        functions = [lambda x, y: x + y,
                     lambda x, y: x - y,
                     lambda x, y: x * y,
                     lambda x, y: x / y
                    ]
        for func in functions:
            ta = torch.tensor(a, requires_grad=True)
            tb = torch.tensor(b, requires_grad=True)
            tc = func(ta, tb).sum()
            ma = Tensor(a.copy(), requires_grad=True)
            mb = Tensor(b.copy(), requires_grad=True)
            mc = func(ma, mb).sum()
            assert arrayEqual(mc.value, tc.detach().numpy())
            tc.backward()
            mc.backward()
            assert arrayEqual(ma.grad, ta.grad.numpy())
            assert arrayEqual(mb.grad, tb.grad.numpy())

    def broadcastTest2():
        a = np.array([3,4], dtype=np.float64)
        b = 3
        functions = [lambda x, y: x + y,
                     lambda x, y: x - y,
                     lambda x, y: x * y,
                     lambda x, y: x / y
                    ]
        for func in functions:
            ta = torch.tensor(a, requires_grad=True)
            tb = b
            tc = func(ta, tb).sum()
            ma = Tensor(a.copy(), requires_grad=True)
            mb = b
            mc = func(ma, mb).sum()
            assert arrayEqual(mc.value, tc.detach().numpy())
            tc.backward()
            mc.backward()
            assert arrayEqual(ma.grad, ta.grad.numpy())

    def reverseOperationTest():
        a = np.array([3,4], dtype=np.float64)
        b = 1
        functions = [lambda x, y: y + x,
                     lambda x, y: y - x,
                     lambda x, y: y * x,
                     lambda x, y: y / x
                    ]
        for func in functions:
            ta = torch.tensor(a, requires_grad=True)
            tb = b
            tc = func(ta, tb).sum()
            ma = Tensor(a.copy(), requires_grad=True)
            mb = b
            mc = func(ma, mb).sum()
            assert arrayEqual(mc.value, tc.detach().numpy())
            tc.backward()
            mc.backward()
            assert arrayEqual(ma.grad, ta.grad.numpy())

    addTest()
    subTest()
    mulTest()
    divTest()
    matmulTest1()
    matmulTest2()
    expTest()
    logTest()
    sumTest()
    transposeTest()
    broadcastUnitTest()
    broadcastTest1()
    broadcastTest2()
    reverseOperationTest()
