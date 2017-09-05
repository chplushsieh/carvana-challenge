
import util.const as const
import util.load as load

def compute_avg():
    train_img_names = load.load_train_imageset()

    for img_name in train_img_names:
        img = load_train_image(const.TRAIN_DIR, img_name)
        # TODO
