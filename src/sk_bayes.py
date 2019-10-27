# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import sys
from sklearn.naive_bayes import GaussianNB
from numpy import array as np_array

DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'


def get_data_from_files(name, fn):
    res = []
    with open(name) as f:
        f.readline()  # ignore header
        for line in f:
            res.append(fn(line))
    return res


def get_value_from_file(name_data):
    data = get_data_from_files(name_data,
                               lambda x: [float(i) for i in x.split(DELI)])
    return data


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


def get_label(sep='', i=''):
    # print(FOLDER+LABEL_FILE+sep+str(i)+EXT)
    label = get_data_from_files(FOLDER+LABEL_FILE+sep+str(i)+EXT, float)
    return label, label


def get_predicted(sep='', i=''):
    # print(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
    #       FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT)
    label = get_data_from_files(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
                                float)
    predicted = get_data_from_files(FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT,
                                    float)
    return label, predicted


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
            data = get_value_from_file(FOLDER+DATA_FILE)
            bayes(get_label, data)
        elif sys.argv[1] == 'train':
            data = get_value_from_file(FOLDER+LEARN+DATA_FILE)
            data_train = get_value_from_file(FOLDER+TEST+DATA_FILE)
            bayes(get_predicted, data, data_train)
    else:
        exit('wrong number of argument')
