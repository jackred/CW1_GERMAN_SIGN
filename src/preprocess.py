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


def split_data(data, train_size=0.7, shuffle=False, random_state=0):
    data_train, data_test = train_test_split(data,
                                             shuffle=shuffle,
                                             random_state=random_state,
                                             train_size=train_size)
    return data_train, data_test


def resize_img(row, d):
    dim = int(len(row) ** (1/2))
    img = row.reshape(dim, dim, 1)
    return (cv2.resize(img, d)).flatten()


def resize_img_square(row, l):
    return resize_img(row, (l, l))


def resize_batch(b, d):
    return np.array([resize_img_square(i, d) for i in b])


def mean_image(label, data):
    return [x.sum(axis=0) / len(x) for x in
            [data[label == i] for i in np.unique(label)]]


def old_segment(image, n_c):
    kmeans = KMeans(n_clusters=n_c, random_state=0).fit(image.reshape(-1, 1))
    return kmeans.labels_ * (255/n_c)


def old_segment_images(rows, n_c):
    return np.array([old_segment(i, n_c) for i in rows])


# def segment(image, n_t):
#     pass
