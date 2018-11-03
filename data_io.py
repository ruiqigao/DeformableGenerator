from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import scipy.misc
from sklearn.externals.joblib import Parallel, delayed
import skimage
from skimage import io
from six.moves import xrange
import tensorflow as tf

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class DataSet(object):
    def __init__(self, data_path, image_size=128, batch_sz=100, prefetch=True, read_len=500, is_mnist=False):
        self.is_mnist = is_mnist
        if is_mnist:
            self.image_size = image_size
            self.batch_sz = batch_sz
            self.num_batch = int(math.ceil(55000.0 / batch_sz))
            self.batch_idx = 0
            self.prefetch = True
        else:
            self.root_dir = data_path
            self.imgList = [f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
            self.imgList.sort()
            self.image_size, self.data_path = image_size, data_path
            self.batch_sz, self.read_len = batch_sz, read_len
            self.num_batch = int(math.ceil(float(len(self.imgList)) / batch_sz))
            self.batch_idx = 0
            self.prefetch = prefetch

        # self.images = np.zeros((len(self.imgList), image_size, image_size, 3)).astype(float)
        if self.prefetch is True:
            self.read_len = self.num_batch
            self._load_data()

    def get_batch(self):
        if self.prefetch is False and np.mod(self.batch_idx, self.read_len) == 0:
            self._load_data()

        start_idx = self.batch_idx * self.batch_sz
        burst_len = self.batch_sz * self.read_len
        end_idx = min((self.batch_idx+1) * self.batch_sz, len(self))
        start_idx_in_batch = np.mod(start_idx, burst_len)
        end_idx_in_batch = np.mod(end_idx, burst_len)
        if end_idx_in_batch == 0:
            end_idx_in_batch = burst_len
        batch_images = self.images[start_idx_in_batch: end_idx_in_batch]
        self.batch_idx = self.batch_idx + 1 if end_idx < len(self) else 0

        return batch_images

    def mean(self):
        return np.mean(self.images, axis=(0, 1, 2, 3))

    def _load_data(self):
        if self.is_mnist:
            print('Loading mnist dataset ...')
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            data = mnist.train.images
            data = data.astype(np.float32)
            data_len = data.shape[0]
            data = np.reshape(data, [-1, 28, 28, 1])
            if not self.image_size == 28:
                data_resize = np.zeros([len(data), self.image_size, self.image_size, 1], dtype=np.float32)
                for i in range(len(data)):
                    data_resize[i, :, :, 0] = scipy.misc.imresize(np.squeeze(data[i, :, :, 0]), [self.image_size, self.image_size])
            else:
                data_resize = data

            self.images = data_resize * 2.0 - 1.0
            print('Data loaded, shape: {}'.format(self.images.shape))
        else:
            start_idx = self.batch_idx * self.batch_sz
            burst_len = self.batch_sz * self.read_len
            end_idx = min(start_idx + burst_len, len(self))
            print('Loading dataset from {} to {}: {}'.format(start_idx, end_idx, self.data_path))
            files = self.imgList[start_idx: end_idx]
            images = self._par_imread(files)
            self.images = np.array(images).astype(np.float32)
            print('Data loaded, shape: {}'.format(self.images.shape))

    def _imread_file(self, file_name):
        file_path = os.path.join(self.data_path, file_name)
        img = np.array(scipy.misc.imresize(skimage.io.imread(file_path), [self.image_size, self.image_size]))
        if len(img.shape) < 3 or img.shape[2] != 3:
            print(img.shape)
            img = img[..., 1]
            img = np.tile(img[..., None], [1, 1, 3])
        max_val = img.max()
        min_val = img.min()
        img = (img - min_val) / (max_val - min_val)
        if img.shape != (self.image_size, self.image_size, 3):
            print('Image {} size {} is not equal to {}, replace with an empty image'.format(file_name, img.shape, (
            self.image_size, self.image_size, 3)))
            img = np.zeros([self.image_size, self.image_size, 3]).astype(np.uint8)
        return img

    def _par_imread(self, files):
        images = []
        for i in xrange(len(files)):
            images.append(self._imread_file(files[i]))
        return images

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.imgList) if self.is_mnist is False else 55000


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images


def clip_by_value(input_, min=0, max=1):
    return np.minimum(max, np.maximum(min, input_))


def img2cell(images, row_num=10, col_num=10, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, images.shape[-1]))
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = clip_by_value(np.squeeze(images[i]), 0, 1)
        temp = temp * 255
        temp = clip_by_value(np.round(temp), min=0, max=255)
        if len(temp.shape) == 3:
            gLow = np.min(temp, axis=(0, 1, 2))
            gHigh = np.max(temp, axis=(0, 1, 2))
        elif len(temp.shape) == 2:
            gLow = np.min(temp, axis=(0, 1))
            gHigh = np.max(temp, axis=(0, 1))
        temp = (temp - gLow) / (gHigh - gLow)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=2)
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
        (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic, :] = temp
    return cell_image


def saveSampleResults(sample_results, filename, col_num=10, margin_syn=2):
    cell_image = img2cell(sample_results, col_num, col_num, margin_syn)
    scipy.misc.imsave(filename, np.squeeze(cell_image))