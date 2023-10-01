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

import cv2
from utils.img_utils import update_hex_image,update_wb_values
from binascii import hexlify, unhexlify
import argparse
from glob import glob
import os
from pipeline.pipeline_utils import get_metadata
import copy

"""
Package clean RAW image (averaged from 30 ISO 50 frames) into a DNG. 
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dng_folder_path', type=str,
                        help='path to day-to-night nighttime dataset iso50 dngs',
                        )
    parser.add_argument('--raw_folder_path', type=str,
                        help='path to day-to-night nighttime averaged clean raw images',
                        )
    parser.add_argument('--save_path', type=str,
                        help='path to save to',
                        default='neural_isp_expts/data/clean_raw_dngs'
                        )
    parser.add_argument('--container_dng_path', type=str,
                        help='container dng',
                        default='assets/container_dngs/container_dng_S20_FE_main_rectilinear_OFF_noise_OFF.dng'
                        )
    parser.add_argument('--wb_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=50816
                        )
    parser.add_argument('--image_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=59288
                        )

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':

    args = parse_args()
    assert os.path.exists(args.save_path), f'{args.save_path} does not exist!'

    dnglist = sorted(glob(os.path.join(args.dng_folder_path,'*.dng')))
    rawlist = sorted(glob(os.path.join(args.raw_folder_path,'*.png')))

    with open(args.container_dng_path, "rb") as fn:
        myhexc = hexlify(fn.read())
    myhexc = bytearray(myhexc)

    for i in range(len(dnglist)):
        print(os.path.basename(dnglist[i]),os.path.basename(rawlist[i]))
        bayer = cv2.imread(rawlist[i],-1)
        metadata = get_metadata(dnglist[i])

        myhex = copy.deepcopy(myhexc)

        wb_dng = metadata['as_shot_neutral']

        myhex = update_wb_values(myhex, wb_dng, args.wb_start)
        myhex = update_hex_image(myhex,bayer,args.image_start)

        db = unhexlify(myhex)

        savepath = os.path.join(args.save_path,os.path.basename(dnglist[i]))
        with open(savepath,"wb") as fb:
            fb.write(db)

    print('Done')