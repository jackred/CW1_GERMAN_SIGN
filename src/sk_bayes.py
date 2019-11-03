# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import argparse
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, \
    MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from helper import get_label, get_data_value, SEP

L = 8
NB = {
    'gnb': {'fn': GaussianNB, 'name': 'Gaussian Naive Bayes'},
    'bnb': {'fn': BernoulliNB, 'name': 'Bernoulli Naive Bayes'},  # useless
    'cnb': {'fn': ComplementNB, 'name': 'Gaussian Naive Bayes'},
    'mnb': {'fn': MultinomialNB, 'name': 'Gaussian Naive Bayes'}
}


def print_line_matrix(lng):
    print('-' * ((L+1) * (lng+1) + 1))


def format_string(a):
    return str(a)[:L].center(L)


def format_row(l):
    return '|'.join([format_string(i) for i in l])


def print_matrix(m, lb):
    print_line_matrix(len(lb))
    print('|' + format_string('lb\pr') + '|'+format_row(lb)+'|')
    print_line_matrix(len(lb))
    for i in range(len(m)):
        print('|' + format_string(lb[i]) + '|' + format_row(m[i]) + '|')
        print_line_matrix(len(lb))


def sk_bayes(fn, data, label, split, shuffle):
    nb = fn()
    rand = np.random.randint(10000000)
    data_train, data_test = train_test_split(data,
                                             shuffle=shuffle,
                                             random_state=rand,
                                             test_size=split)
    label_train, label_test = train_test_split(label,
                                               shuffle=shuffle,
                                               random_state=rand,
                                               test_size=split)
    y_predicted = nb.fit(data_train, label_train).predict(data_test)
    lb = np.unique(label)
    matrix = confusion_matrix(label_test, y_predicted)
    print_matrix(matrix, lb)


def bayes(name_nb, fn_label, data, split, shuffle):
    label = fn_label()
    print('using all class')
    print('****%s****' % NB[name_nb]['name'])
    sk_bayes(NB[name_nb]['fn'], data, label, split, shuffle)
    for i in range(10):
        print('=======')
        label = fn_label(sep=SEP, i=i)
        print('class %d' % i)
        sk_bayes(NB[name_nb]['fn'], data, label, split, shuffle)


def parse_args():
    argp = argparse.ArgumentParser('sklearn bayes')
    argp.add_argument('-r', dest='randomize', default=False,
                      action='store_true')
    argp.add_argument('-s', dest='split', type=float, default=1)
    argp.add_argument('-d', dest='data', default='')
    argp.add_argument('-l', dest='label', default='')
    argp.add_argument('-f', dest='folder', default='')
    argp.add_argument('-b', dest='bayes', required=True)
    return argp.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = get_data_value(name=args.folder + args.data)
    bayes(name_nb=args.bayes,
          fn_label=lambda sep='', i='':
          get_label(sep=sep, i=i, name=args.folder + args.label),
          data=data,
          split=args.split,
          shuffle=args.randomize)
