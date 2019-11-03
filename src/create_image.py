# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import sys
import os
from helper import get_data_raw, DELI

if len(sys.argv) > 3:
    sys.exit('wrong argument')

if len(sys.argv) >= 2:
    if sys.argv[1].isdigit():
        nb = int(sys.argv[1])
    else:
        exit('not a number')
else:
    nb = 50

name = ''
if len(sys.argv) == 3:
    name = sys.argv[2]

data = get_data_raw(name=name)

if 'test' not in os.listdir('..'):
    os.mkdir('../test')

for j in range(nb):
    with open('../test/test' + str(j) + '.ppm', 'w+') as f:
        d = data[j].replace(DELI, '\n')
        d = d.replace('.0', '')
        f.write('P2\n48 48 255\n')
        f.write(d)
