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

Convert PS tif to png
"""

import argparse
import cv2
from utils.general_utils import check_dir
import os
from glob import glob
from shutil import rmtree

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tif_dir', default='neural_isp_expts/data/clean_srgb_tiffs/', type=str, help='tif dir')
    parser.add_argument('--save_dir', default='real_dataset/clean', type=str, help='save dir')

    args = parser.parse_args()

    # remove png files already inside
    if os.path.isdir(args.save_dir): rmtree(args.save_dir)

    # create directory again
    check_dir(args.save_dir)

    allfiles = sorted(glob(os.path.join(args.tif_dir, '*.tif')))

    for fil in allfiles:
        cleanimg = cv2.imread(os.path.join(fil), cv2.IMREAD_UNCHANGED)
        destination = os.path.join(args.save_dir, os.path.basename(fil)[:-4] + '.png')
        cv2.imwrite(destination, cleanimg)

