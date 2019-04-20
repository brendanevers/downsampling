import multiprocessing
import numpy as np
import math
from pprint import pprint


def mode(array):  # It turns out that numpy does not have a built-in mode function, which I did not know prior to this
    unique, counts = np.unique(array, return_counts=True)
    stats = dict(zip(unique, counts))
    return max(stats, key=stats.get)


def splitter(array, block):  # Make array into array of blocks
    # Create shape tuple for shape of new array. First tuple is the shape of the array of blocks, second is blocks
    shape = tuple(np.array(array.shape) // block) + tuple(block)
    # Get strides for new array. First is strides of outer array, then inner
    strides = tuple(array.strides * block) + array.strides
    # Get our new array
    out = np.lib.stride_tricks.as_strided(array, shape, strides)
    return out


class Main:  # We start off by making a class in order to make the starting array
    #  a class variable that we can pass between various functions
    def __init__(self, initial_array):
        if type(initial_array) == list:
            self.initial_array = np.array(initial_array)
        elif type(initial_array) == np.ndarray:
            self.initial_array = initial_array
        else:
            raise TypeError('Argument must be a list or a numpy array')
        for item in np.ndenumerate(self.initial_array):
            try:
                int(self.initial_array[item[0]])
            except:
                raise ValueError('Array elements must be real numbers')
        for size in self.initial_array.shape:
            if math.log(size, 2) % 1 != 0:
                raise ValueError('Array or list must have dimensions 2^L')
            else:
                continue

    def downsize(self, l):  # This is the down sample function. It returns the down sample for the array and a given l
        #  Create the block array with blocks of shape 2^l in each dimension of the array
        blocked = splitter(self.initial_array, np.array([2 ** l] * self.initial_array.ndim))

        out = np.zeros(blocked.shape[:len(self.initial_array.shape)])  # Dummy array to receive modes
        for item in np.ndenumerate(out):  # Gets indices for out
            # takes the mode of each index in blocked and adds it to the same place in out
            out[item[0]] = mode(blocked[item[0]])
        pprint(out)

    def main(self):
        pprint(self.initial_array)
        task = [i for i in range(1, int(math.log(min(self.initial_array.shape), 2)) + 1)]  # array to accumulate all l's
        p = multiprocessing.Pool(multiprocessing.cpu_count())  # Creates pool of processes equal to number of
        #  CPUs in order to bypass Python's global interpreter lock
        p.map(self.downsize, task)  # Maps each downsize(l) to a process


def go(array):  # Call creates an object around the array then runs main, as a way to facilitate automatic testing
    Main(array).main()


if __name__ == '__main__':
    # A number of arrays to test. Unused arrays are commented out to save memory
    test_array_1 = np.array(([False, 1, 1, 1, 1, 1, 1, 1],[1, 2, 1, 2, 1, 2, 1, 2],[1, 1, 2, 2, 2, 2, 2, 2],[1, 2, 2, 2, 2, 2, 2, 2]))
    test_array_2 = [[1, 2], [3, 4]]
    test_array_3 = [[[1, 2], [3, 4]], [[2, 6], [7, 8]]]
    test_array_4 = 4  # For exception handling
    test_array_5 = np.random.random_sample(size=(512, 512, 16))
    test_array_6 = ['a']
    #  Create an object around the array, then run the main function
    input_array = test_array_1
    go(input_array)
