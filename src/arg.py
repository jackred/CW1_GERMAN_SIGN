# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import argparse


def float_or_int(i):
    n = float(i)
    if n >= 1:
        return int(i)
    elif n > 0 and n < 1:
        return n
    else:
        msg = 'Not a float in range(0,1) or a integer > 0'
        raise argparse.ArgumentTypeError(msg)


FILTER_LIST = ['s', 'r', 'p', 'c', 'm', 'g']
SEGMENT_LIST = ['w', 's', 'f']
TH_OPTION = 't'
CONTRAST_LIST = ['e', 'm', 'a', 'l']


def choice_segment(c):
    if len(c) > 2:
        msg = 'too much argument for segment'
        raise argparse.ArgumentTypeError(msg)
    elif len(c) == 2:
        if c[0] in SEGMENT_LIST and c[1] == TH_OPTION:
            return (c[0], c[1])
        else:
            msg = 'First arg must be ' + str(SEGMENT_LIST) + \
                ' and second an interger > 0'
            raise argparse.ArgumentTypeError(msg)
    elif len(c) == 1:
        if c in SEGMENT_LIST:
            return c
        else:
            msg = 'Arg must be ' + str(SEGMENT_LIST)
            raise argparse.ArgumentTypeError(msg)
    else:
        msg = 'No arg provided'
        raise argparse.ArgumentTypeError(msg)


def contrast_arg(c):
    if len(c) >= 2:
        if c[0] in CONTRAST_LIST and c[1:].isdigit() and int(c[1:]) > 0:
            return (c[0], int(c[1:]))
        else:
            msg = 'First arg must be ' + str(CONTRAST_LIST) + \
                ' and second t'
            raise argparse.ArgumentTypeError(msg)
    else:
        msg = 'No enough arg provided'
        raise argparse.ArgumentTypeError(msg)


def parse_args(name):
    argp = argparse.ArgumentParser(name)
    argp.add_argument('-r', dest='randomize', default=False,
                      action='store_true')
    argp.add_argument('-s', dest='split', type=float_or_int)
    argp.add_argument('-k', dest='kmeans', type=int)
    argp.add_argument('-d', dest='data')
    argp.add_argument('-l', dest='label')
    argp.add_argument('-f', dest='folder')
    argp.add_argument('-c', dest='contrast', type=contrast_arg)
    argp.add_argument('-z', dest='size', type=int)
    argp.add_argument('-p', dest='pooling', type=int)
    argp.add_argument('-hi', dest='histogram', default=False,
                      action='store_true')
    argp.add_argument('-eh', dest='equalize', default=False,
                      action='store_true')
    argp.add_argument('-bin', dest='binarise', default=False,
                      action='store_true')
    argp.add_argument('-fi', dest='filters', choices=FILTER_LIST)
    argp.add_argument('-g', dest='segment', type=choice_segment)
    return argp


BAYES_LIST = ['gnb', 'cnb', 'bnb', 'mnb']


def bayes_args():
    argp = parse_args('sk_learn naive bayes')
    argp.add_argument('-b', dest='bayes', required=True,
                      choices=BAYES_LIST)
    argp.add_argument('-cm', dest='create_mean', default=False,
                      action='store_true')
    return argp.parse_args()


def kmeans_args():
    argp = parse_args('sk_learn kmeans')
    argp.add_argument('-i', dest='init', default='k',
                      help='k: kmeans++ | m: mean | r: random')
    argp.add_argument('-cm', dest='create_mean', default=False,
                      action='store_true')
    return argp.parse_args()


def bayes_net_args():
    argp = parse_args('bayes network')
    return argp.parse_args()


def preprocess_args():
    argp = parse_args('preprocessing')
    argp.add_argument('-n', dest='name', default='')
    argp.add_argument('-m', dest='mean', default=False, action='store_true')
    return argp.parse_args()
