import util.ensemble as ensemble
import util.submit as submit
import util.const as const

import rle_loader

def apply_rle(rle_loader):
    img_rles = {}

    for i, (img_name, rle) in enumerate(rle_loader):
        assert len(img_name) == 1
        assert len(rle) == 1

        img_name = img_name[0]
        rle = rle[0]

        img_rles[img_name] = rle

    # save submission.csv
    submit.save_predictions(const.ENSEMBLE_DIR_NAME, img_rles)
    return


if __name__ == "__main__":

    exp_names = ensemble.get_models_ensembled()
    print('The predictions are ensemble by {}. '.format(exp_names))

    # TODO print the augmentations as along as the models 

    rle_loader = rle_loader.get_rle_loader()

    apply_rle(rle_loader)
