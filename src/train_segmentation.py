# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import helper
import argparse
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.color import label2rgb
from skimage.filters import sobel
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def watershed_g(img):
    gradient = sobel(img)
    labels = watershed(gradient)
    return labels


choice = {'s': slic, 'f': felzenszwalb, 'w': watershed_g}


def parse_args():
    argp = argparse.ArgumentParser('train tg')
    argp.add_argument('-t', dest='train', default='s', choices=choice.keys())
    return argp.parse_args()


def sk_bayes(data_train, label_train, data_test, label_test):
    nb = MultinomialNB()
    y_predicted = nb.fit(data_train, label_train).predict(data_test)
    good_prediction = (((y_predicted == label_test).sum()) / len(label_test))
    return good_prediction * 100


def main():
    data = helper.get_data()
    dim = int(len(data[0]) ** (1/2))
    data = data.reshape(len(data), dim, dim)
    label = helper.get_label()
    fn = choice[parse_args().train]
    return data, label, fn


def fitness(data, label, fn, kwargs={}):
    img = np.array([fn(i, **kwargs).flatten() for i in data])
    res = sk_bayes(img, label, img, label)
    return res
