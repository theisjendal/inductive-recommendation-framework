import random


class SeedGenerator:
    """
    Used to get seed for concurrent processes, see https://stackoverflow.com/a/31058798/13598523
    """
    def __init__(self, seed):
        self._rand = random.Random(seed)

    def get_seed(self):
        return self._rand.randint(0, 255)
