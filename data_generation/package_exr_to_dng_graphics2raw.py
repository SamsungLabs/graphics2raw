"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Abhijith Punnappurath (abhijith.p@samsung.com)
Luxi Zhao (lucy.zhao@samsung.com)

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
import scipy.io

from pipeline.pipeline_utils import get_metadata
from utils.img_utils import *
import copy
import pickle

RAND_SEED = 101
np.random.seed(RAND_SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exr_folder_path', type=str,
                        help='path to exr images')  # expected directory structure: exr_folder_path/<scene name>/<GT_xx_xx>/xxx.exr
    parser.add_argument('--save_path', type=str,
                        help='path to save to',
                        default='neural_isp_expts/data/graphics_dngs_graphics2raw')
    parser.add_argument('--container_dng_path', type=str,
                        help='path to container dng',
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
    parser.add_argument('--train_val_set', type=str,
                        help='create dngs only for these images',
                        default='assets/split_files/graphics2raw_train_val_list.p'
                        )
    parser.add_argument('--no_safe_invert', action='store_true',
                        help='Do not use safe invert for highlight regions, setting this flag will make the saturated regions have a color cast',
                        )
    parser.add_argument('--rgb_gain', action='store_true',
                        help='Apply random global gain to images',
                        )
    parser.add_argument('--rgb_gain_mean', type=float, default=0.8,
                        help='Mean value for rgb gain, used only when rgb_gain is set to True.',
                        )
    parser.add_argument('--max_illums', type=int, default=None,
                        help='Maximum number of illuminants used to build the convex hull. None for no max limit',
                        )
    parser.add_argument('--illum_seed', type=int, default=None,
                        help='Random seed for sampling illuminants used to build the convex hull.',
                        )
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':

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

    metadata = get_metadata(args.container_dng_path)
    metadata = get_extra_tags(args.container_dng_path, metadata)

    # IMPORTANT: for S20FE, color_matrix_1 corresponds to D65,
    # color_matrix_2 corresponds to Standard Light A
    # The order may be different for different cameras, need to check beforehand!
    cmD65 = metadata['color_matrix_1']
    cmA = metadata['color_matrix_2']

    # Illumination dictionary
    # load nighttime illuminants
    gt_illum = scipy.io.loadmat('./data_generation/night_dict_v2.mat')
    gt_illum = gt_illum['gt_illum']
    gt_illum = gt_illum[0:45, :]
    # discard outliers
    indd = gt_illum[:, 0] < 1
    gt_illum = gt_illum[indd, :]

    if args.max_illums is not None:
        # Sample N illums from 39 illums
        np.random.seed(args.illum_seed)
        rand_idx = np.random.choice(len(gt_illum), args.max_illums, replace=False)
        gt_illum = gt_illum[rand_idx, :]
        np.random.seed(RAND_SEED)  # reset the seed

    print(gt_illum.shape[0], 'num illums')

    gt_illum[:, 0], gt_illum[:, 1], gt_illum[:, 2] = get_illum_normalized_by_g(gt_illum)

    illum_mean = np.mean(gt_illum, 0)
    illum_cov = np.cov(np.transpose(gt_illum))

    # for S20FE main camera
    white_level = 1023
    black_level = 64

    w = 4032
    h = 3024

    with open(args.container_dng_path, "rb") as fn:
        myhexc = hexlify(fn.read())
    myhexc = bytearray(myhexc)

    xyz2srgb_mat = get_xyz_to_srgb_mat()

    for i in range(len(exrlist)):
        pathnames = exrlist[i][0].split('/')
        savename = pathnames[-3] + '_' + pathnames[-1]

        if savename[:-4] in train_val_set:
            print(exrlist[i])

            exr_img = cv2.imread(exrlist[i][0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            exr_img = exr_img[:, :, ::-1]

            # Resize to the shape of S20 FE RAW image: 3024 x 4032
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

            # Sample an illuminant
            while True:
                wb_vec = np.random.multivariate_normal(illum_mean, illum_cov, 1).squeeze()
                if in_hull(np.expand_dims(wb_vec[[0, 2]], axis=0), gt_illum[:, [0, 2]]):
                    break
            print(wb_vec)

            # Compute CST matrix
            cst_mat = get_cst_matrix(cmD65, cmA, wb_vec)

            # sRGB to CIE XYZ (Invert XYZ2sRGB)
            raw_est = apply_combined_mat(exr_img, np.linalg.inv(xyz2srgb_mat))

            # CIE XYZ to device RGB (Invert CST)
            raw_est = apply_combined_mat(raw_est, np.linalg.inv(cst_mat))

            # Inverse digital gain (optional)
            rgb_gain = np.random.normal(loc=args.rgb_gain_mean, scale=0.1) if args.rgb_gain else 1.0

            # Invert WB
            if not args.no_safe_invert:
                raw_est = safe_invert_gains(raw_est, wb_vec, rgb_gain)
            else:
                wb_mat = get_wb_as_matrix(wb_vec)
                raw_est *= rgb_gain
                raw_est = apply_combined_mat(raw_est, np.linalg.inv(wb_mat))

            wb_dng = wb_vec

            # Mosaic
            raw_est = RGB2bayer(raw_est)  # Bayer for S20 FE: G R B G

            # Denormalize
            raw_est = raw_est * (white_level - black_level) + black_level
            raw_est[raw_est < 0] = 0
            raw_est[raw_est > white_level] = white_level

            # Save to DNG
            myhex = copy.deepcopy(myhexc)
            myhex = update_wb_values(myhex, wb_dng, args.wb_start)
            myhex = update_hex_image(myhex, raw_est, args.image_start)

            db = unhexlify(myhex)

            savepath = os.path.join(args.save_path, savename[:-4] + '.dng')
            with open(savepath, "wb") as fb:
                fb.write(db)

    print('Done')
