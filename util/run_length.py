
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
    #code = None
    rle= None
    mask=np.transpose(mask)
    
    inds = mask.flatten()
    head=inds[0]
    inds[0]=0
    tail=inds[inds.size-1]
    print(tail)
    inds[inds.size-1]=0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    
    rle=''
    if head==1:
        if inds[1]==1:
            runs[0]=1
            runs[1]+=1
        else :
            rle='1 1 '
        
    if tail==1:
        if inds[inds.size-2]==1:
            runs[runs.size-1]+=1
        else :
            runs=np.append(runs,[1,1])    
    
    rle = rle + ' '.join([str(r) for r in runs])    
    
    return rle


#def stringify_code(code):
#    '''
#    input:
#      code: a list of tuples representing encoded input mask using run length encoding
#            Note that it may not be sorted
#            For example: [(4, 2), (1, 1)]
#    output:
#      stringified: a string which is stringified code in sorted order
#                   For example: '1 1 4 2'
#    '''
#    output = None
#    # TODO
#    return output

if __name__ == "__main__":

    test_arr = np.array([
        [1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ])

#    encoded_output = [(1, 1), (3, 1), (5, 2), (10, 4)]
#    assert encoded_output == run_length_encode(test_arr)

    stringified_output = '1 1 3 1 5 2 10 4'
    assert stringified_output == run_length_encode(test_arr)#stringify_code(encoded_output)
