import numpy as np
import numbers

def random_index_based_on_weights(weights, random_state):
    """ random_index_based_on_weights
    Generates a random index, based on index weights and a random
    generator instance.
    Parameters
    ----------
    weights: list
        The weights of the centroid's indexes.
    random_state: numpy.random
        A random generator.
    Returns
    -------
    int
        The generated index.
    """
    prob_sum = np.sum(weights)
    val = random_state.rand() * prob_sum
    index = 0
    sum_value = 0.0
    while (sum_value <= val) & (index < len(weights)):
        sum_value += weights[index]
        index += 1
    return index - 1

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Notes
    -----
    Code from sklearn
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance'.format(seed))
