from typing import List

import numpy as np
import random

from shared.seed_generator import SeedGenerator
from shared.user import User


def get_ratings_matrix(users: List[User]):
    ratings = [[user.index, item, rating] for user in users for item, rating in user.ratings]

    return np.array(ratings)


def set_seeds(sg: SeedGenerator):
    random.seed(sg.get_seed())
    np.random.seed(sg.get_seed())
