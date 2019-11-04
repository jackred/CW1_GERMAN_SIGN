# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
import helper

DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'


def get_labels(sep='', i='', name='', folder=FOLDER, label_file=LABEL_FILE,
               ext=EXT):
    label = helper.get_label(sep, i, name, folder, label_file, ext)
    return label, label


def get_predicted(sep='', i=''):
    # print(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
    #       FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT)
    label = helper.get_data_from_files(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
                                       int)
    predicted = helper.get_data_from_files(
        FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT,
        int)
    return label, predicted


def get_data_and_train(folder=FOLDER, data_file=DATA_FILE, deli=DELI,
                       learn=LEARN, test=TEST):
    data = helper.get_value_from_file(folder+learn+data_file, deli)
    data_train = helper.get_value_from_file(folder+test+data_file, deli)
    return data, data_train
