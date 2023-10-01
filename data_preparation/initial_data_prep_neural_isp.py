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

Input is two folders: one containing DNGs and other containing tiffs
Output folders: (to match current day-to-night code)
1) train : clean, clean_raw, metadata_raw, noisy_raw
2) val : clean, clean_raw, metadata_raw, noisy_raw
"""

import argparse
import cv2
import pickle
from utils.general_utils import check_dir
import os
from glob import glob
from shutil import copyfile, rmtree
from pipeline.pipeline_utils import get_metadata, get_visible_raw_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dng_dir', type=str, help='dng dir')
    parser.add_argument('--tif_dir', type=str, help='tif dir')
    parser.add_argument('--val_names', default='Living,Playground,Rustic', type=str, help='name of test dataset')
    parser.add_argument('--save_dir', default='graphics_dataset', type=str, help='save dir')
    parser.add_argument('--noisy', action='store_true', help='noisy dataset')

    args = parser.parse_args()

    if os.path.isdir(args.save_dir): rmtree(args.save_dir)
      
    print(args.dng_dir)
    print(args.tif_dir)

    # create directories
    check_dir(args.save_dir)
    for fol in ['train', 'val']:  check_dir(os.path.join(args.save_dir, fol))
    for fol in ['train', 'val']:
        subfols = ['clean_raw', 'clean', 'metadata_raw']
        if args.noisy:
            subfols.append('noisy_raw')
        for subfol in subfols: check_dir(os.path.join(args.save_dir, fol, subfol))

    allfiles = [os.path.basename(x) for x in sorted(glob(os.path.join(args.dng_dir, '*.dng')))]

    valnames = [item for item in args.val_names.split(',')]

    index = {'train': [i for i in range(len(allfiles)) if allfiles[i].split('_')[0] not in valnames],
             'val': [i for i in range(len(allfiles)) if allfiles[i].split('_')[0] in valnames]}

    input_dir_dng = args.dng_dir
    input_dir_tif = args.tif_dir

    for fol in ['train', 'val']:
        for ind in index[fol]:
            for subfol in ['clean']:
                cleanimg = cv2.imread(os.path.join(input_dir_tif, allfiles[ind][:-4] + '.tif'), cv2.IMREAD_UNCHANGED)
                destination = os.path.join(args.save_dir, fol, subfol, allfiles[ind][:-4] + '.png')
                cv2.imwrite(destination, cleanimg)

            for subfol in ['clean_raw']:
                cleanrawimg = get_visible_raw_image(os.path.join(input_dir_dng, allfiles[ind]))
                destination = os.path.join(args.save_dir, fol, subfol, allfiles[ind][:-4] + '.png')
                cv2.imwrite(destination, cleanrawimg)
        
            if args.noisy:
                for subfol in ['noisy_raw']:
                    source = os.path.join(input_dir_dng, 'noisy_raw', allfiles[ind][:-4] + '.png')
                    destination = os.path.join(args.save_dir, fol, subfol, allfiles[ind][:-4] + '.png')
                    copyfile(source,destination)

            for subfol in ['metadata_raw']:
                metadata = get_metadata(os.path.join(input_dir_dng, allfiles[ind]))
                pickle.dump(metadata, open(os.path.join(args.save_dir, fol, subfol, allfiles[ind][:-4] + '.p'), "wb"))
