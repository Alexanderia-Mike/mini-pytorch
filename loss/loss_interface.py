from model.tensor import Tensor

class LossInterface:
    def __call__(self, predicted:Tensor, target:Tensor) -> Tensor:
        raise NotImplementedError