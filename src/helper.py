# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

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
    return res


def get_value_from_file(name_data, deli):
    data = get_data_from_files(name_data,
                               lambda x: [float(i) for i in x.split(deli)])
    return data


def get_label(sep='', i=''):
    # print(FOLDER+LABEL_FILE+sep+str(i)+EXT)
    label = get_data_from_files(FOLDER+LABEL_FILE+sep+str(i)+EXT, float)
    return label


def get_labels(sep='', i=''):
    label = get_label(sep, i)
    return label, label


def get_predicted(sep='', i=''):
    # print(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
    #       FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT)
    label = get_data_from_files(FOLDER+LEARN+LABEL_FILE+sep+str(i)+EXT,
                                float)
    predicted = get_data_from_files(FOLDER+TEST+LABEL_FILE+sep+str(i)+EXT,
                                    float)
    return label, predicted


def get_data():
    return get_value_from_file(FOLDER+DATA_FILE, DELI)


def get_data_and_train():
    data = get_value_from_file(FOLDER+LEARN+DATA_FILE, DELI)
    data_train = get_value_from_file(FOLDER+TEST+DATA_FILE, DELI)
    return data, data_train
