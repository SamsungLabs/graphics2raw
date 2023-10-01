"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Abhijith Punnappurath (abhijith.p@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

from shutil import copyfile, rmtree
import os
from glob import glob
import argparse
import cv2
import pickle

from pipeline.pipeline_utils import get_metadata, get_visible_raw_image
from utils.general_utils import check_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='real_dataset', type=str,
                        help='base address')
    parser.add_argument('--save_path', default='real_dataset_k_fold', type=str,
                        help='output address')
    parser.add_argument('--which_fold', default=0, type=int, help='which fold for testing [0,1,2]')
    parser.add_argument('--kfold_indices', default='assets/split_files/day2night_k_fold_indices.p', type=str,
                        help='load saved k-fold indices')
    parser.add_argument('--with_noise', default=0, type=int, help='for noisy case')
    parser.add_argument('--only_iso_3200', action='store_true',
                        help='True: only run on iso_3200.')
    parser.add_argument('--only_iso_1600', action='store_true',
                        help='True: only run on iso_1600.')
    parser.add_argument('--test_on_val', default=0, type=int, help='for testing on validation set')

    args = parser.parse_args()

    if args.only_iso_3200:
        assert not args.only_iso_1600
    elif args.only_iso_1600:
        assert not args.only_iso_3200
    # if they are both false, mix iso 1600 and 3200 equally
    print(args)

    return args


def split_func(args, folder_path, k_fold_indices):
    save_path = args.save_path

    train_index = k_fold_indices['train_index_all'][args.which_fold]
    val_index = k_fold_indices['val_index_all'][args.which_fold]
    test_index = k_fold_indices['test_index_all'][args.which_fold]

    if not args.only_iso_3200 and not args.only_iso_1600:
        if folder_path == 'iso_1600':
            index = {'train': train_index[0::2], 'val': val_index[0::2]}
        elif folder_path == 'iso_3200':
            index = {'train': train_index[1::2], 'val': val_index[1::2]}
        else:
            index = {'train': train_index, 'val': val_index}
    else:
        index = {'train': train_index, 'val': val_index}

    input_dir = args.base_path

    print(index['train'])
    print('...')
    print(index['val'])
    print('...')
    print(test_index)
    print('...')
    print(args.which_fold, len(index['train']), len(index['val']), len(test_index))

    # create directories if they don't exist
    check_dir(save_path)
    for fol in ['train', 'val', 'test']:  check_dir(os.path.join(save_path, fol))
    for fol in ['train', 'val']:
        for subfol in ['clean_raw', 'noisy_raw', 'clean', 'metadata_raw']: check_dir(os.path.join(save_path, fol, subfol))
    for fol in ['test']:
        for subfol in ['clean_raw', 'clean', 'dng']: check_dir(os.path.join(save_path, fol, subfol))
    check_dir(os.path.join(save_path, 'test', 'dng', folder_path))
    if args.test_on_val:
        check_dir(os.path.join(save_path, 'val', 'dng', folder_path))

    allfiles = [os.path.basename(x) for x in sorted(glob(os.path.join(input_dir, 'clean', '*.png')))]

    for fol in ['train', 'val']:
        for ind in index[fol]:
            for subfol in ['clean_raw', 'clean']:
                source = os.path.join(input_dir, subfol, allfiles[ind])
                destination = os.path.join(save_path, fol, subfol, allfiles[ind])
                copyfile(source, destination)

            for subfol in ['noisy_raw']:
                rawimg = get_visible_raw_image(os.path.join(input_dir, 'dng', folder_path, allfiles[ind][:-4] + '.dng'))
                destination = os.path.join(save_path, fol, subfol, allfiles[ind])
                cv2.imwrite(destination, rawimg)

            for subfol in ['metadata_raw']:
                metadata = get_metadata(os.path.join(input_dir, 'dng', folder_path, allfiles[ind][:-4] + '.dng'))
                pickle.dump(metadata, open(os.path.join(save_path, fol, subfol, allfiles[ind][:-4] + '.p'), "wb"))

            if fol == 'val' and args.test_on_val:
                print('saving val dngs')
                source = os.path.join(input_dir, 'dng', folder_path, allfiles[ind][:-4] + '.dng')
                destination = os.path.join(save_path, fol, 'dng', folder_path, allfiles[ind][:-4] + '.dng')
                copyfile(source, destination)

    for fol in ['test']:
        clean_raw_dir = os.listdir(os.path.join(save_path, fol, 'clean_raw'))
        clean_dirs_empty = len(clean_raw_dir) == 0
        for ind in test_index:
            if clean_dirs_empty:
                for subfol in ['clean_raw', 'clean']:
                    source = os.path.join(input_dir, subfol, allfiles[ind])
                    destination = os.path.join(save_path, fol, subfol, allfiles[ind])
                    copyfile(source, destination)

            for subfol in ['dng']:
                source = os.path.join(input_dir, subfol, folder_path, allfiles[ind][:-4] + '.dng')
                destination = os.path.join(save_path, fol, subfol, folder_path, allfiles[ind][:-4] + '.dng')
                copyfile(source, destination)


if __name__ == "__main__":
    args = parse_args()
    k_fold_indices = pickle.load(open(args.kfold_indices, "rb"))

    if os.path.isdir(args.save_path): rmtree(args.save_path)

    if args.with_noise:
        if args.only_iso_3200:
            split_func(args, 'iso_3200', k_fold_indices)
        elif args.only_iso_1600:
            split_func(args, 'iso_1600', k_fold_indices)
        else:
            split_func(args, 'iso_1600', k_fold_indices)
            split_func(args, 'iso_3200', k_fold_indices)
    else:
        split_func(args, 'iso_50', k_fold_indices)

    print('Done!')
