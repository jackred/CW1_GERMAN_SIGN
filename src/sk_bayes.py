# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from arg import bayes_args
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, \
    MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np
from helper import pre_processed_data, pre_processed_label, SEP
from preprocess import split_data

L = 8
NB = {
    'gnb': {'fn': GaussianNB, 'name': 'Gaussian Naive Bayes'},
    'bnb': {'fn': BernoulliNB, 'name': 'Bernoulli Naive Bayes'},  # useless
    'cnb': {'fn': ComplementNB, 'name': 'Gaussian Naive Bayes'},
    'mnb': {'fn': MultinomialNB, 'name': 'Gaussian Naive Bayes'}
}


def print_line_matrix(lng):
    print('-' * ((L+1) * (lng+2) + 1))


def format_string(a):
    return str(a)[:L].center(L)


def format_row(l):
    return '|'.join([format_string(i) for i in l])


def print_matrix(m, lb):
    print_line_matrix(len(lb))
    print('|' + format_string('lb\pr') + '|' + format_row(lb) + '|'
          + format_string('total') + '|')
    print_line_matrix(len(lb))
    for i in range(len(m)):
        print('|' + format_string(lb[i]) + '|' + format_row(m[i]) + '|'
              + format_string(sum(m[i])) + '|')
        print_line_matrix(len(lb))
    print('|' + format_string('total') + '|'
          + format_row(sum(m)) + '|'
          + format_string(m.sum()) + '|')
    print_line_matrix(len(lb))


def sk_bayes(fn, data_train, label_train, data_test, label_test):
    nb = fn()
    y_predicted = nb.fit(data_train, label_train).predict(data_test)
    lb = np.unique(label_train)
    matrix = confusion_matrix(label_test, y_predicted)
    print_matrix(matrix, lb)


def bayes(name_nb, fn_label, data_train, data_test):
    label_train, label_test = fn_label()
    print('using all class')
    print('****%s****' % NB[name_nb]['name'])
    sk_bayes(NB[name_nb]['fn'], data_train, label_train, data_test, label_test)
    for i in range(10):
        print('=======')
        label_train, label_test = fn_label(sep=SEP, i=i)
        print('class %d' % i)
        sk_bayes(NB[name_nb]['fn'], data_train, label_train, data_test,
                 label_test)


if __name__ == '__main__':
    args = bayes_args()
    rand = np.random.randint(10000000)
    data_train, data_test = pre_processed_data(args, rand)
    bayes(name_nb=args.bayes,
          fn_label=lambda sep='', i='':
          pre_processed_label(option=args, rand=rand,
                              sep=sep, i=i),
          data_train=data_train,
          data_test=data_test)
