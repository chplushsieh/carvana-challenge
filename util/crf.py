
import numpy as np

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


def apply_crf(masks, probs):
    '''
    input:
      masks: pytorch Variable in gpu

    modified from:
    https://github.com/yt605155624/tensorflow-deeplab-resnet/blob/e81482d7bb1ae674f07eae32b0953fe09ff1c9d1/inference_crf.py
    '''

    unary = softmax_to_unary(probs)
    unary = np.ascontiguousarray(unary)

    # TODO
    num_classes = 1 # or 2?
    # img.shape = ?

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], num_classes)
    d.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res
