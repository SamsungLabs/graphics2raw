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

Package graphics exr image to DNG file
"""

import os
import argparse
from glob import glob
from binascii import unhexlify

import tensorflow as tf

from utils.img_utils import *
import copy
from data_generation.unprocess import unprocess
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exr_folder_path', type=str,
                        help='path to exr images')  # expected directory structure exr_folder_path/<scene name>/<GT_xx_xx>/xxx.exr
    parser.add_argument('--save_path', type=str,
                        help='path to save to',
                        default='./neural_isp_expts/data/graphics_dngs_upi')
    parser.add_argument('--container_dng_path', type=str,
                        help='path to save to',
                        default='assets/container_dngs/container_dng_S20_FE_main_rectilinear_OFF_gain_OFF_noise_OFF_cam_calib_OFF.dng'
                        )
    parser.add_argument('--wb_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=50816
                        )
    parser.add_argument('--image_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=59288
                        )
    parser.add_argument('--colormatrix1_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=50240
                        )
    parser.add_argument('--colormatrix2_start', type=int,
                        help='magic value for S20 FE main camera',
                        default=50384
                        )
    parser.add_argument('--train_val_set', type=str,
                        help='create dngs only for these images',
                        default='assets/split_files/graphics2raw_train_val_list.p'
                        )

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    RAND_SEED = 101
    np.random.seed(RAND_SEED)
    tf.random.set_seed(RAND_SEED)

    args = parse_args()
    assert os.path.exists(args.save_path), f'{args.save_path} does not exist!'

    exrlist = []

    allsubfol = [f.path for f in os.scandir(args.exr_folder_path) if f.is_dir()]
    for subfol in allsubfol:
        allsubsubfol = [f.path for f in os.scandir(os.path.join(args.exr_folder_path, subfol)) if f.is_dir()]
        for subsubfol in allsubsubfol:
            exrlistfol = sorted(glob(os.path.join(args.exr_folder_path, subfol, subsubfol, '*.exr')))
            exrlist.append(exrlistfol)

    train_val_set = pickle.load(open(args.train_val_set, "rb"))

    # for S20 FE main camera
    white_level = 1023
    black_level = 64

    w = 4032
    h = 3024

    with open(args.container_dng_path, "rb") as fn:
        myhexc = hexlify(fn.read())
    myhexc = bytearray(myhexc)

    for i in range(len(exrlist)):
        pathnames = exrlist[i][0].split('/')
        savename = pathnames[-3] + '_' + pathnames[-1]
        
        
        if savename[:-4] in train_val_set:
            print(exrlist[i])

            exr_img = cv2.imread(exrlist[i][0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # .astype('float32')
            exr_img = exr_img[:, :, ::-1]  # .exr image is BGR, change it to RGB for processing

            # resize to 3024 x 4032
            if exr_img.shape[0] < h and exr_img.shape[1] < w:
                exr_img = cv2.resize(exr_img, dsize=(w, h))
            elif exr_img.shape[0] < h and exr_img.shape[1] > w:
                exr_img = exr_img[:, 0:w]
                exr_img = cv2.resize(exr_img, dsize=(w, h))
            elif exr_img.shape[0] > h and exr_img.shape[1] < w:
                exr_img = exr_img[0:h, :]
                exr_img = cv2.resize(exr_img, dsize=(w, h))
            else:
                exr_img = exr_img[0:h, 0:w]

            exr_img = np.clip(exr_img, 0, 1)
            exr_img = exr_img ** (1/2.2)
            exr_img = tf.convert_to_tensor(exr_img, dtype=tf.float32)

            raw_est, metadata = unprocess(exr_img)
            raw_est = raw_est.numpy().astype(np.float32)
            for k, v in metadata.items():
                metadata[k] = v.numpy().astype(np.float32)

            wb_dng = np.array([1 / metadata['red_gain'], 1, 1 / metadata['blue_gain']])

            # Bayer for S20 FE
            # GR
            # BG
            raw_est = RGB2bayer(raw_est)

            # denormalize
            raw_est = raw_est * (white_level - black_level) + black_level
            raw_est[raw_est < 0] = 0
            raw_est[raw_est > white_level] = white_level

            colormatrix1= metadata['xyz2cam'].flatten().astype(np.float64)

            myhex = copy.deepcopy(myhexc)
            myhex = update_wb_values(myhex, wb_dng, args.wb_start)
            myhex = update_hex_image(myhex, raw_est, args.image_start)
            myhex = update_colormatrix1_values(myhex, colormatrix1, args.colormatrix1_start)
            myhex = update_colormatrix1_values(myhex, colormatrix1, args.colormatrix2_start) # repeat for cm2

            db = unhexlify(myhex)

            savepath = os.path.join(args.save_path, savename[:-4] + '.dng')
            with open(savepath, "wb") as fb:                
                fb.write(db)

    print('Done')
