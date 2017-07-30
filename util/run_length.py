
import numpy as np

'''
Carvana competition uses Run Length encoding to reduce the size of submission:
https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
'''


# TODO Modify the following encoder for our use:
# https://www.kaggle.com/stainsby/fast-tested-rle
# or
# https://www.kaggle.com/hackerpoet/even-faster-srun-length-encoder

def run_length_encode(mask):
    '''
    input:
      mask: a numpy array with only 0's or 1's in it
            For example: np.array([[0, 1], [0, 1]])
    output:
      code: a list of tuples representing encoded input mask using run length encoding
            For example: [(3, 2)]
    '''
    code = None
    # TODO
    return code

def stringify_code(code):
    '''
    input:
      code: a list of tuples representing encoded input mask using run length encoding
            Note that it may not be sorted
            For example: [(4, 2), (1, 1)]
    output:
      stringified: a string which is stringified code in sorted order
                   For example: '1 1 4 2'
    '''
    output = None
    # TODO
    return output

if __name__ == "__main__":

    test_arr = np.array([
        [1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ])

    encoded_output = [(1, 1), (3, 1), (5, 2), (10, 4)]
    assert encoded_output == run_length_encode(test_arr)

    stringified_output = '1 1 3 1 5 2 10 4'
    assert stringified_output == stringify_code(encoded_output)
