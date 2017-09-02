import os.path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
import csv



def rgb_shift(img):
    mu = 0
    sigma = 0.003
    evals = np.array([  7.88291483e+00,   3.93729159e+01,   1.04797824e+04])
    evecs = np.array([[ 0.24347043, -0.77981712, -0.57672125],
                    [-0.79493354,  0.18024043, -0.5793048 ],
                    [ 0.55570029,  0.59949866, -0.57601957]])

    feature_vec=np.matrix(evecs)

    # 3 x 1 scaled eigenvalue matrix
    se = np.zeros((3,1))
    se[0][0] = np.random.normal(mu, sigma)*evals[0]
    se[1][0] = np.random.normal(mu, sigma)*evals[1]
    se[2][0] = np.random.normal(mu, sigma)*evals[2]
    se = np.matrix(se)
    val = feature_vec*se
    # print(se, evals, val)

    img_pca=np.zeros((1280,1918,3))
    for k in range(img.shape[2]):
        img_pca[:,:,k]=np.matrix.__add__(img[:,:,k], val[k])
        img_pca[img_pca[:,:,k]>255]=255
        img_pca[img_pca[:,:,k]<0]=0

    return img_pca.astype('uint8')


if __name__ == "__main__":

    def load_imageset(imageset_path):
        img_names = []
        with open(imageset_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                img_names.append('/home/judichunt/git/carvana-challenge/data/train/' + row[0] + '.jpg')

        img_names.sort()
        return img_names

    DATA_DIR = '/home/judichunt/git/carvana-challenge/data/'
    TRAIN_IMAGESET_PATH = os.path.join(DATA_DIR, 'train.csv')

    img_names = load_imageset(TRAIN_IMAGESET_PATH)

    imlist = (io.imread_collection(img_names))

    img = imlist[0]
    im = rgb_shift(img)
    plt.imshow(im)
    plt.show()