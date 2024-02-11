import random

class XavierInitializer:
    def __init__(self, fan_in:int) -> None:
        self.fan_in = fan_in

    def __call__(self) -> float:
        return random.normalvariate(sigma=1./self.fan_in)