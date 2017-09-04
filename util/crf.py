
import numpy as np

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


def apply_crf(masks, probs):
    '''
    input:
      masks: pytorch Variable in gpu

    Modified from:
    https://github.com/yt605155624/tensorflow-deeplab-resnet/blob/e81482d7bb1ae674f07eae32b0953fe09ff1c9d1/inference_crf.py

    Usage:
      masks = crf.apply_crf(masks, outputs)
    '''

    unary = softmax_to_unary(probs)
    unary = np.ascontiguousarray(unary)

    # TODO
    num_classes = 1 # or 2?
    # img.shape = ?

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], num_classes)
    d.setUnaryEnergy(unary)
    # TODO unary protential is not dependent on the image itself?

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # TODO this potential is not dependent on the image itself?

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    Q = d.inference(5)  # set the number of iterations
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res
