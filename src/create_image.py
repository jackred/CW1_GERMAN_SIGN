# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import csv
import sys
from os import mkdir

# IMG_SIZE = 48


if (len(sys.argv) == 1):
    sys.exit('toto')


with open(sys.argv[1]) as csvf:
    sr = csv.reader(csvf, delimiter=',')
    imgs = []
    for row in sr:
        imgs.append(row)

imgs.pop()

mkdir('test')

for j in range(len(imgs)):
    with open('./test/test' + str(j) + '.ppm', 'w+') as f:
        f.write('P3\n48 48 255\n')
        f.write('\n'.join([str(int(float(i))) + ' ' + str(int(float(i))) + ' '
                           + str(int(float(i))) for i in imgs[j]]))
