# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import sys
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, \
    MultinomialNB
from numpy import unique as np_unique,  \
    logical_and as np_logical_and
from helper import get_labels, get_predicted, get_data, get_data_and_train, SEP

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
    total = np_predicted.size
    a_s_p_i = total - s_p_i
    # a_s_y_i = total - s_y_i
    true_positive = np_logical_and(predicted_i, y_predicted_i).sum()
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


def sk_bayes(fn, data, label, data_train=None, predicted=None):
    nb = fn()
    y_predicted = nb.fit(data, label).predict(data_train)
    mislabeled = (predicted != y_predicted).sum()
    print('accuracy: %.3f%%' % ((1 - (mislabeled / len(data_train))) * 100))
    lb = np_unique(label)
    make_matrix(predicted, y_predicted, lb)


def bayes(name_nb, fn_label, data, data_train=None):
    if data_train is None:
        data_train = data
    label, predicted = fn_label()
    print('using all class')
    print('****%s****' % NB[name_nb]['name'])
    sk_bayes(NB[name_nb]['fn'], data, label, data_train, predicted)
    for i in range(10):
        print('=======')
        label, predicted = fn_label(sep=SEP, i=i)
        print('class %d' % i)
        sk_bayes(NB[name_nb]['fn'], data, label, data_train, predicted)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        if sys.argv[2] in NB:
            if sys.argv[1] == 'test':
                data = get_data()
                bayes(sys.argv[2], get_labels, data)
            elif sys.argv[1] == 'train':
                data, data_train = get_data_and_train()
                bayes(sys.argv[2], get_predicted, data, data_train)
        else:
            exit('wrong argument, list is: %s' % ', '.join(NB))
    else:
        exit('wrong number of argument')
