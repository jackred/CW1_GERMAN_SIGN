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


def parse_args(name):
    argp = argparse.ArgumentParser(name)
    argp.add_argument('-r', dest='randomize', default=False,
                      action='store_true')
    argp.add_argument('-s', dest='split', type=float_or_int)
    argp.add_argument('-g', dest='segment', type=int)
    argp.add_argument('-d', dest='data')
    argp.add_argument('-l', dest='label')
    argp.add_argument('-f', dest='folder')
    argp.add_argument('-z', dest='size', type=int)
    argp.add_argument('-hi', dest='histogram', default=False,
                      action='store_true')
    return argp


def bayes_args():
    argp = parse_args('sk_learn naive bayes')
    argp.add_argument('-b', dest='bayes', required=True)
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
