import numpy as np
from typing import Callable

class Predecessor:
    def __init__(self, predecessor, transformer:Callable) -> None:
        self.predecessor = predecessor
        self.transformer = transformer