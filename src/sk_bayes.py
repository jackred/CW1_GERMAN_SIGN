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
import numpy as np
from helper import get_label, get_data_value, SEP

L = 8
NB = {
    'gnb': {'fn': GaussianNB, 'name': 'Gaussian Naive Bayes'},
    'bnb': {'fn': BernoulliNB, 'name': 'Bernoulli Naive Bayes'},  # useless
    'cnb': {'fn': ComplementNB, 'name': 'Gaussian Naive Bayes'},
    'mnb': {'fn': MultinomialNB, 'name': 'Gaussian Naive Bayes'}
}


def print_line_matrix():
    print('-' * (8 * 4 + 5))


def format_string(a):
    return str(a)[:L].center(L)


def print_row_matrix(a, b, c, d):
    print('|%s|%s|%s|%s|' % (format_string(a),
                             format_string(b),
                             format_string(c),
                             format_string(d)))


def print_matrix(i, fp, tp, fn, tn):
    print_line_matrix()
    print_row_matrix(i, 'FALSE', 'TRUE', '')
    print_line_matrix()
    print_row_matrix('Pos', fp, tp, fp+tp)
    print_line_matrix()
    print_row_matrix('Neg', fn, tn, fn+tn)
    print_line_matrix()
    print_row_matrix('', fp+fn, tp+tn, fp+tn+tp+fn)
    print_line_matrix()


def make_one_matrix(np_predicted, y_predicted, lb, i):
    predicted_i = np_predicted == i
    y_predicted_i = y_predicted == i
    s_p_i = predicted_i.sum()
    s_y_i = y_predicted_i.sum()
    total = len(np_predicted)
    a_s_p_i = total - s_p_i
    # a_s_y_i = total - s_y_i
    true_positive = np.logical_and(predicted_i, y_predicted_i).sum()
    false_positive = s_y_i - true_positive
    false_negative = s_p_i - true_positive
    true_negative = a_s_p_i - false_positive
    print_matrix(i, false_positive, true_positive, false_negative,
                 true_negative)


def make_matrix(np_predicted, y_predicted, lb):
    if len(lb) == 2:
        make_one_matrix(np_predicted, y_predicted, lb, 0)
    else:
        for i in lb:
            make_one_matrix(np_predicted, y_predicted, lb, i)


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
    mislabeled = (label_test != y_predicted).sum()
    print('accuracy: %.3f%%' % ((1 - (mislabeled / len(data_train))) * 100))
    lb = np.unique(label)
    make_matrix(label_test, y_predicted, lb)


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
