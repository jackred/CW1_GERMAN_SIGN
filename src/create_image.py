# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import sys
import os
from helper import get_data_raw, from_csv_to_ppm_raw, create_image

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
    d = from_csv_to_ppm_raw(data[j])
    create_image('../test/test' + str(j) + '.ppm', d)
