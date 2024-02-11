class OptimInterface:
    def step(self) -> None:
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        if "parameters" not in self.__dict__:
            raise Exception("parameters not set for optimizer!")
        for parameter in self.parameters:
            parameter.zero_grad()
