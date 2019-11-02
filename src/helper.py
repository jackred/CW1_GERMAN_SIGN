# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np


DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'


def get_data_from_files(name, fn):
    res = []
    with open(name) as f:
        f.readline()  # ignore header
        for line in f:
            res.append(fn(line))
    return np.array(res)


def get_value_from_file(name_data, deli):
    data = get_data_from_files(name_data,
                               lambda x: [float(i) for i in x.split(deli)])
    return data


def get_label(sep='', i='', folder=FOLDER, label_file=LABEL_FILE, ext=EXT):
    # print(folder+label_file+sep+str(i)+EXT)
    label = get_data_from_files(folder+label_file+sep+str(i)+ext, int)
    return label


def get_labels(sep='', i='', folder=FOLDER, label_file=LABEL_FILE, ext=EXT):
    label = get_label(sep, i, folder, label_file, ext)
    return label, label


def get_predicted(sep='', i=''):
    # print(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
    #       FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT)
    label = get_data_from_files(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
                                float)
    predicted = get_data_from_files(FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT,
                                    float)
    return label, predicted


def get_data_value(name='', folder=FOLDER, data_file=DATA_FILE, deli=DELI):
    return get_value_from_file(name or folder+data_file, deli)


def get_data_raw(name='', folder=FOLDER, data_file=DATA_FILE):
    return get_data_from_files(name or folder+data_file,
                               lambda x: x)


def get_data_and_train(folder=FOLDER, data_file=DATA_FILE, deli=DELI,
                       learn=LEARN, test=TEST):
    data = get_value_from_file(folder+learn+data_file, deli)
    data_train = get_value_from_file(folder+test+data_file, deli)
    return data, data_train
