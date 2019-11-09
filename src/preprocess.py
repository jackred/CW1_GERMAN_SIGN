# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.exposure import match_histograms, equalize_hist
from skimage.transform import resize
from skimage.filters import threshold_isodata, sobel, roberts, scharr, \
    prewitt, gaussian, median
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.color import label2rgb, rgb2gray  # , gray2rgb
from skimage.future.graph import rag_mean_color, cut_threshold


FILTER = {'s': sobel, 'r': roberts, 'p': prewitt, 'c': scharr, 'm': median,
          'g': gaussian}


def split_data(data, train_size=0.7):
    data_train, data_test = train_test_split(data,
                                             shuffle=False,
                                             train_size=train_size)
    return data_train, data_test


def randomize(data, rand):
    np.random.seed(rand)
    np.random.shuffle(data)
    return data


def new_resize_img(row, d):
    dim = int(len(row) ** (1/2))
    img = row.reshape(dim, dim)
    return resize(img, d).flatten()


def resize_img(row, d):
    dim = int(len(row) ** (1/2))
    img = row.reshape(dim, dim, 1)
    return cv2.resize(img, d).flatten()


def resize_img_square(row, l):
    return resize_img(row, (l, l))


def resize_batch(b, d):
    return np.array([resize_img_square(i, d) for i in b])


def mean_image(label, data):
    return [x.sum(axis=0) / len(x) for x in
            [data[label == i] for i in np.unique(label)]]


def mean_images_global(data):
    return data.mean(0)


def old_segment(image, n_c):
    kmeans = KMeans(n_clusters=n_c, random_state=0).fit(image.reshape(-1, 1))
    return kmeans.labels_ * (255/(n_c-1))


def old_segment_images(rows, n_c):
    return np.array([old_segment(i, n_c) for i in rows])


def adjust_histogram(img, mean):
    return match_histograms(img, mean)


def adjust_histograms(data):
    mean = mean_images_global(data)
    return np.array([adjust_histogram(i, mean) for i in data])


def equalize_histogram(img):
    return equalize_hist(img)*255


def equalize_histograms(data):
    return np.array([equalize_hist(i) for i in data])


def binarise(img):
    th = threshold_isodata(img)
    return (img > th) * 255


def binarise_images(data):
    return np.array([binarise(i) for i in data])


def watershed_g(img):
    gradient = sobel(img)
    labels = watershed(gradient)
    return labels


def cut_thr(img, labels, n=10):
    g = rag_mean_color(img, labels)
    labels2 = cut_threshold(labels, g, n)
    return labels2


SEGMENT = {'s': slic, 'w': watershed_g, 'f': felzenszwalb, 't': cut_thr}


def segment(img, s):
    dim = int(len(img) ** (1/2))
    img = img.reshape(dim, dim)
    labels = SEGMENT[s[0]](img)
    if len(s) == 1:
        labels = cut_thr(img, labels)
    return rgb2gray(label2rgb(labels, img, kind='avg')).flatten()


def segment_images(data, s):
    return np.array([segment(i, s) for i in data])


def filters(img, ed):
    dim = int(len(img) ** (1/2))
    img = img.reshape(dim, dim)
    return FILTER[ed](img).flatten()


def filter_images(data, ed):
        return np.array([filters(i, ed) for i in data])
