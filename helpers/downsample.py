from ast import Num
from typing import Tuple
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import trange
import numba


@numba.jit(nopython=True)
def softmax(x: np.array) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def intersect_effective_size_and_counts(
    counts: pd.DataFrame, effective_length: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Intersects the effective size and the counts.
    :param counts: The counts to be intersected.
    :param effective_length: The effective length to be intersected.
    :return: The intersected counts and effective length.
    """
    counts = counts.copy()
    effective_length = effective_length.copy()
    # intersect the effective length with the counts
    counts = counts.loc[effective_length.index]
    effective_length = effective_length.loc[counts.index]
    return counts, effective_length



def get_probability(x: np.array) -> np.array:
    """
    classic probability
    """
    x = x.copy()
    x[x < 0] = 0
    return x / x.sum()


def make_sure_numpy_array_positive(array: np.array) -> np.array:
    """
    Makes sure that all elements of an array are positive.
    :param array: The array to be modified.
    :return: The modified array.
    """
    array[array < 0] = 0
    return array


# @numba.jit(nopython=True)
def random_subtract_from_array(
    array: np.array, rate: float, seed: int = 42, chunk: Num = 1000
) -> np.array:
    """
    Subtracts a random number from each element of an array.
    Make resulted array proportional to the original one, by rate provided
    :param array: The array to be modified.
    :param rate: The rate of the subtraction.
    :return: The modified array.
    """
    # set random seed
    np.random.seed(seed)
    # downsample rate to size
    downsample_total = array.sum() * rate
    # subtract only the elements that are greater than 0
    # for i in trange(int(downsample_total / chunk)):
    for i in range(int(downsample_total / chunk)):
        # from positive position get chunk size of coordinates
        array = make_sure_numpy_array_positive(array)
        positive_array = array[array > 0]
        downsample_position = np.random.choice(
            np.where(array > 0)[0], size=chunk, p=get_probability(positive_array), replace=True
        )
        count_downsample = Counter(downsample_position)
        array[list(count_downsample.keys())] -= list(count_downsample.values())

    return array


def downsample(
    counts: pd.DataFrame, rate: float, seed: int = 42, chunk: Num = 100
) -> pd.DataFrame:
    """
    Subtracts a random number from each element of an array.
    Make resulted array proportional to the original one, by rate provided
    :param array: The array to be modified.
    :param rate: The rate of the subtraction.
    :return: The modified array.
    """
    # for each column, do the subtraction
    counts = counts.copy()
    for column in counts.columns:
        counts[column] = random_subtract_from_array(
            counts[column].values.astype(np.float32), rate, seed, chunk
        )
        # make sure that all elements are positive
        counts[column] = make_sure_numpy_array_positive(counts[column].values)
    return counts
