import numpy as np


class BaseTracker:
    def track(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Trying to call track method from the abstract class")
