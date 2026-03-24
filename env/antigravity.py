import numpy as np

class GravityManager:
    """Manages the dynamic gravity field vectors."""
    def __init__(self):
        # Possible gravity vectors: (dx, dy)
        self.vectors = [
            (0, 1),   # Down
            (0, -1),  # Up
            (1, 0),   # Right
            (-1, 0),  # Left
            (0, 0)    # Zero-G
        ]
        self.current_gravity = (0, 0)
        
    def shift(self):
        """Randomly selects a new gravity vector."""
        idx = np.random.randint(0, len(self.vectors))
        self.current_gravity = self.vectors[idx]
        return self.current_gravity

    def get_gravity_name(self):
        names = {
            (0, 1): "DOWN",
            (0, -1): "UP",
            (1, 0): "RIGHT",
            (-1, 0): "LEFT",
            (0, 0): "ZERO-G"
        }
        return names.get(self.current_gravity, "UNKNOWN")
