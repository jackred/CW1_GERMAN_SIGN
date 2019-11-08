# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from arg import kmeans_args
from sklearn.cluster import KMeans
from helper import pre_processed_data, pre_processed_label, matrix_confusion, \
    create_images_from_rows
import numpy as np
from preprocess import mean_image


def main():
    args = kmeans_args()
    rand = np.random.randint(10000000)
    data_train, data_test = pre_processed_data(args, rand)
    label_train, label_test = pre_processed_label(args, rand)
    print('data loaded')
    switch_choice = {'k': lambda: 'k-means++', 'r': lambda: 'random',
                     'm': lambda: mean_image(label_train, data_train)}
    kmeans = KMeans(n_clusters=10, random_state=0,
                    init=switch_choice[args.init]()).fit(data_train)
    predicted = kmeans.predict(data_test)
    print('kmeans done')
    compare_class(predicted, label_test)
    if args.create_mean:
        create_images_from_rows('km', kmeans.cluster_centers_)


def compare_class(predicted, label):
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    found = dict(zip(unique_p, counts_p))
    unique_l, counts_l = np.unique(label, return_counts=True)
    label_nb = dict(zip(unique_l, counts_l))
    print('found: ', found)
    print('label: ', label_nb)
    matrix_confusion(label, predicted, unique_l)
    # for j in range(0, len(unique_l)):
    #     predicted = (predicted + 1) % len(unique_l)
    #     matrix_confusion(label, predicted, unique_l)


if __name__ == '__main__':
    main()
