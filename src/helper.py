# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
import preprocess
from sklearn.metrics import confusion_matrix


DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'
L = 8


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


# give label as numpy array of integer
def get_label(sep='', i='', name='', folder=FOLDER, label_file=LABEL_FILE,
              ext=EXT):
    # print(folder+label_file+sep+str(i)+EXT)
    label = get_data_from_files((name or folder+label_file)+sep+str(i)+ext,
                                int)
    return label


# give data as numpy array of integer
def get_data_value(name='', folder=FOLDER, data_file=DATA_FILE, deli=DELI):
    return get_value_from_file(name or folder+data_file, deli)


# give data as numpy of string (one cell = one row)
def get_data_raw(name='', folder=FOLDER, data_file=DATA_FILE):
    return get_data_from_files(name or folder+data_file,
                               lambda x: x)


def create_image(name, d, w, h):
    with open(name, 'w+') as f:
        f.write('P2\n%d %d 255\n' % (w, h))
        f.write(d)


# convert an array of integer to a ppm stirng
# ex: [1.0, 2.0, 9.4] => '1\n2\n9\n'
def convert_ppm_raw(row):
    return '\n'.join([str(int(round(i))) for i in row])+'\n'


# convert csv data to ppm string
# ex: '1.0,5.0,255.0' => '1\n5\n255'
def from_csv_to_ppm_raw(row):
    return row.replace(DELI, '\n').replace('.0', '')+'\n'


# create an image of name 'name' (extension must be wrote)
# format ppm 2 (P2)
# from the row given
def create_image_from_row(name, row):
    s = convert_ppm_raw(row)
    wh = len(row) ** (1/2)
    create_image(name, s, wh, wh)


def pre_processed_file(file_value, option, rand=0):
    if option.split is not None:
        file_value, file_value_test = preprocess.split_data(file_value,
                                                            option.split,
                                                            option.randomize,
                                                            rand)
    else:
        file_value_test = file_value
    return file_value, file_value_test


def pre_processed_data(option, rand):
    data = get_data_value(name=option.folder + option.data)
    if option.size is not None:
        data = preprocess.resize_batch(data, option.size)
    return pre_processed_file(data, option)


def pre_processed_label(option, rand, sep='', i=''):
    label = get_label(sep=sep, i=i, name=option.folder + option.label)
    return pre_processed_file(label, option)


def print_line_matrix(lng):
    print('-' * ((L+1) * (lng+2) + 1))


def format_string(a):
    return str(a)[:L].center(L)


def format_row(l):
    return '|'.join([format_string(i) for i in l])


def print_matrix(m, lb):
    print_line_matrix(len(lb))
    print('|' + format_string('lb\pr') + '|' + format_row(lb) + '|'
          + format_string('total') + '|')
    print_line_matrix(len(lb))
    for i in range(len(m)):
        print('|' + format_string(lb[i]) + '|' + format_row(m[i]) + '|'
              + format_string(sum(m[i])) + '|')
        print_line_matrix(len(lb))
    print('|' + format_string('total') + '|'
          + format_row(sum(m)) + '|'
          + format_string(m.sum()) + '|')
    print_line_matrix(len(lb))


# create and print confusion_matrix
def matrix_confusion(label, predicted, lb):
    matrix = confusion_matrix(label, predicted)
    print_matrix(matrix, lb)
