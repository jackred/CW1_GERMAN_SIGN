# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from arg import bayes_args
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, \
    MultinomialNB
import numpy as np
from helper import pre_processed_data, pre_processed_label, SEP, \
    matrix_confusion, create_images_from_rows
from preprocess import mean_image

L = 8
NB = {
    'gnb': {'fn': GaussianNB, 'name': 'Gaussian Naive Bayes'},
    'bnb': {'fn': BernoulliNB, 'name': 'Bernoulli Naive Bayes'},  # useless
    'cnb': {'fn': ComplementNB, 'name': 'Gaussian Naive Bayes'},
    'mnb': {'fn': MultinomialNB, 'name': 'Gaussian Naive Bayes'}
}


def sk_bayes(fn, data_train, label_train, data_test, label_test):
    nb = fn()
    y_predicted = nb.fit(data_train, label_train).predict(data_test)
    lb = np.unique(label_train)
    matrix_confusion(label_test, y_predicted, lb)
    return data_test, y_predicted


def bayes(name_nb, fn_label, data_train, data_test, cm):
    label_train, label_test = fn_label()
    print('using all class')
    print('****%s****' % NB[name_nb]['name'])
    res = []
    res.append(sk_bayes(NB[name_nb]['fn'], data_train, label_train, data_test,
                        label_test))
    for i in range(10):
        print('=======')
        label_train, label_test = fn_label(sep=SEP, i=i)
        print('class %d' % i)
        res.append(sk_bayes(NB[name_nb]['fn'], data_train, label_train,
                            data_test, label_test))
    if cm:
        for i in range(len(res)):
            create_images_from_rows('%d_b' % i,
                                    mean_image(res[i][1], res[i][0]))


if __name__ == '__main__':
    args = bayes_args()
    rand = np.random.randint(10000000)
    data_train, data_test = pre_processed_data(args, rand)
    bayes(name_nb=args.bayes,
          fn_label=lambda sep='', i='':
          pre_processed_label(option=args, rand=rand,
                              sep=sep, i=i),
          data_train=data_train,
          data_test=data_test,
          cm=args.create_mean)
