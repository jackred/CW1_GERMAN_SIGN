# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import arg
import helper
import random
import preprocess


def main():
    args = arg.preprocess_args()
    rand = random.randint(0, 9999999)
    data, _ = helper.pre_processed_data(args, rand, dry=False)
    #  data = data.reshape(-1,48,48)
    if args.mean:
        label, _ = helper.pre_processed_label(args, rand, dry=False)
        data = preprocess.mean_image(label, data)
    elif args.split is not None or args.randomize:
        label, _ = helper.pre_processed_label(args, rand, dry=False)
        helper.write_data_to_file((args.folder or helper.FOLDER)
                                  + (args.name or 'processed_data') + '_l'
                                  + helper.EXT,
                                  label,
                                  fmt='%d',
                                  h='0')
    # for i in range(4):
    #     for j in range(4):
    helper.write_data_to_file((args.folder or helper.FOLDER)
                              + (args.name or 'processed_data')  # + str(i) + str(j)
                              + helper.EXT,
                              data,
                              #  [k[i*12:(i*12)+12,j*12:(j*12)+12].flatten() for k in data],
                              h=', '.join(str(i) for i in range(len(data[0]))))


if __name__ == '__main__':
    main()
