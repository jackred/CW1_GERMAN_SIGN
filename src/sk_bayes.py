# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import sys
from sklearn.naive_bayes import GaussianNB
from numpy import array as np_array
from helper import get_labels, \
    get_predicted, get_data, get_data_and_train

DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'


def sk_bayes(data, label, data_train=None, predicted=None):
    data_train = data_train or data
    predicted = predicted or label
    gnb = GaussianNB()
    np_predicted = np_array(predicted)
    y_predicted = gnb.fit(data, label).predict(data_train)
    mislabeled = (np_predicted != y_predicted).sum()
    print('accuracy: %.3f%%' % ((1 - (mislabeled / len(data_train))) * 100))
    print('Number of mislabeled points out of a total %d points: %d'
          % (len(data_train), mislabeled))


def bayes(fn_label, data, data_train=None):
    data_train = data_train or data
    label, predicted = fn_label()
    print('using all class')
    sk_bayes(data, label, data_train, predicted)
    for i in range(10):
        print('=======')
        label, predicted = fn_label(sep=SEP, i=i)
        print('class %d' % i)
        sk_bayes(data, label, data_train, predicted)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
            data = get_data()
            bayes(get_labels, data)
        elif sys.argv[1] == 'train':
            data, data_train = get_data_and_train()
            bayes(get_predicted, data, data_train)
    else:
        exit('wrong number of argument')
