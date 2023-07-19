import numpy as np
from PIL import Image
from imageio.v2 import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os
from codes.utils import *
from torch.utils.data import Dataset
from codes import mvtecad

DATASET_PATH = '/home/son/Work/Anomaly-Detection-PatchSVDD-PyTorch/dataset'

def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))

def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    # print(images.shape)
    return images

def get_mean(obj):
    images = get_x(obj, mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean

def get_x(obj, mode='train'):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/{mode}/*/*.png')
    fpaths = sorted(glob(fpattern))

    if mode == 'test':
        fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
        fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

        images1 = np.asarray(list(map(imread, fpaths1)))
        images2 = np.asarray(list(map(imread, fpaths2)))
        images = np.concatenate([images1, images2])

    else:
        images = np.asarray(list(map(imread, fpaths)))

    if images.shape[-1] != 3:
        images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images)
    return images


def get_x_standardized(obj, mode='train'):
    x = get_x(obj, mode=mode)
    mean = get_mean(obj)
    return (x.astype(np.float32) - mean) / 255

pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}

def generate_coords(H, W, K):
    # h = np.random.randint(0, H - K + 1)
    # w = np.random.randint(0, W - K + 1)
    h=64
    w=64
    return h, w

def generate_coords_position(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    pos = np.random.randint(8)

    with task('P2'):
        J = K // 4 # K:64 -> J:16

        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2, pos

def crop_image_CHW(image, coord, K):
    h, w = coord
    return image[:, h: h + K, w: w + K]

def generate_coords_svdd(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    with task('P2'):
        J = K // 32

        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1)
            w_jit = np.random.randint(-J, J + 1)

        h2 = h1 + h_jit
        w2 = w1 + w_jit

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2

if __name__ == '__main__':
    # train_x = get_x_standardized("bottle", mode='train')
    train_x = get_x("bottle", mode='train')
    train_x = NHWC2NCHW(train_x)

    x = np.asarray(train_x)
    n = 0
    K = 64
    image = x[n]

    datasets = dict()
    # datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
    # datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)

    # image = x[n]
    p1, p2, pos = generate_coords_position(256, 256, 64)
    # p1, p2 = generate_coords_svdd(256, 256, K=64)

    patch1 = crop_image_CHW(image, p1, K).copy()
    patch2 = crop_image_CHW(image, p2, K).copy()

    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)

    # rgbshift1 = np.random.normal(scale=10, size=(3, 1, 1))
    # rgbshift2 = np.random.normal(scale=100, size=(3, 1, 1))

    # patch1 += rgbshift1
    # patch2 += rgbshift2


    # # additive noise
    noise1 = np.random.normal(scale=0.02, size=(3, K, K))
    noise2 = np.random.normal(scale=1, size=(3, K, K))

    patch1 += noise1
    patch2 += noise2

    patch1 = patch1.astype(np.int)
    patch2 = patch2.astype(np.int)

    import matplotlib.pyplot as plt

    print(pos)
    patch1 = np.transpose(patch1, [1, 2, 0])
    patch2 = np.transpose(patch2, [1, 2, 0])
    plt.imshow(patch1)
    plt.show()
    plt.imshow(patch2)
    plt.show()


