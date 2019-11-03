# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

from sklearn.cluster import KMeans
from helper import get_data, get_label
import numpy as np


def main():
    data = get_data()
    print('data loaded')
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    print('kmeans done')
    compare_class(kmeans)


def compare_class(kmeans):
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    found = dict(zip(unique, counts))
    print('found: ', found)
    label = np.array(get_label())
    print('label loaded')
    size_one_class = [1410, 1860, 420, 1320, 2100, 2160, 780, 240, 2070, 300]
    for j in range(0, 9):
        print('\n\n--------\n')
        for i in range(0, 9):
            print('=======================')
            print('kmeans: %d - label: %d' % (j, i))
            kj_class = kmeans.labels_ == j
            li_class = label == i
            compare = np.logical_and(kj_class, li_class).sum()
            percentage = (compare / size_one_class[i]) * 100
            print('good: %d out of %d | accuracy: %.2f%%' %
                  (compare, size_one_class[i], percentage))


if __name__ == '__main__':
    main()
