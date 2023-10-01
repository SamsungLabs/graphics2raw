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
"""

import os
import pickle
import argparse
from utils.img_utils import *
import scipy.io
from data_preparation.data_generator_illum_est import get_gt_illum_by_fname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphics_path', type=str,
                        help='path to graphics images',
                        default='illum_est_expts/synthia/SYNTHIA_RAND_CVPR16/RGB'
                        )
    parser.add_argument('--train_val_set', type=str,
                        help='only use these images for training and validation',
                        default='assets/split_files/illum_est/synthia_train_val_list.p'
                        )
    parser.add_argument('--save_path', type=str,
                        help='path to exr images',
                        default='illum_est_expts/data/SamsungNX2000/ours/'
                        )
    parser.add_argument('--target_camera', type=str,
                        help='use camera name to select CM1 and CM2',
                        default='SamsungNX2000'
                        )
    parser.add_argument('--cm_file', type=str,
                        help='path to NUS CM1 and CM2',
                        default='assets/container_dngs/NUS_CST_mats.p'
                        )
    parser.add_argument('--mat_file', type=str,
                        help='mat file with gt illuminations',
                        default='illum_est_expts/nus_metadata/nus_outdoor_gt_illum_mats/SamsungNX2000_gt.mat'
                        )
    parser.add_argument('--use_train_val_illums_only', action='store_true',
                        help='sample illuminants from the convex hull built from training and validation illuminants only'
                        )
    parser.add_argument('--split_file', type=str,
                        help='path to file specifying which filenames belong to which split, example:path/to/SamsungNX2000_train_valid_test_split_fns.p')
    parser.add_argument('--max_illums', type=int,
                        help='Maximum number of illuminants used to build the convex hull. None for no max limit',
                        default=None)
    parser.add_argument('--no_safe_invert', action='store_true',
                        help='Do not use safe invert for highlight regions, setting this flag will make the saturated regions have a color cast',
                        )
    parser.add_argument('--rgb_gain', action='store_true',
                        help='Apply random global gain to images',
                        )
    parser.add_argument('--rgb_gain_mean', type=float, default=1.0,
                        help='Mean value for rgb gain, used only when rgb_gain is set to True.',
                        )

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':

    RAND_SEED = 101
    np.random.seed(seed=RAND_SEED)

    args = parse_args()

    train_val_set = pickle.load(open(args.train_val_set, 'rb'))

    metadata = pickle.load(open(args.cm_file, 'rb'))[args.target_camera]
    cmD65 = metadata['cm_D65']
    cmA = metadata['cm_A']

    # load illuminants
    if args.use_train_val_illums_only:
        # Use only training and validation illuminants to build the convex hull
        gt_illum_by_fname = get_gt_illum_by_fname(args.mat_file)
        splits = pickle.load(open(args.split_file, 'rb'))['real_split']
        split_fns_train = splits['train']
        split_fns_valid = splits['valid']
        split_fns = np.append(split_fns_train, split_fns_valid)
        print('Number of training and validation illums: ', len(split_fns))

        fnames = [os.path.basename(f)[:-4] for f in split_fns]
        gt_illum = [gt_illum_by_fname[fname] for fname in fnames]
        gt_illum = np.concatenate([gt_illum])  # all 150 train and validation illums

        # sample within the training and validation illums
        if args.max_illums is not None:
            seed = 0
            np.random.seed(seed)
            rand_idx = np.random.choice(len(gt_illum), args.max_illums, replace=False)
            gt_illum = gt_illum[rand_idx, :]
            np.random.seed(RAND_SEED)  # reset the seed
        print('Number of illums used to build the convex hull: ', len(gt_illum))

    else:
        gt_illum = scipy.io.loadmat(args.mat_file)
        gt_illum = gt_illum['groundtruth_illuminants']
        gt_illum[:, 0], gt_illum[:, 1], gt_illum[:, 2] = get_illum_normalized_by_g(gt_illum)
    illum_mean = np.mean(gt_illum, 0)
    illum_cov = np.cov(np.transpose(gt_illum))

    xyz2srgb_mat = get_xyz_to_srgb_mat()

    gt_illum_array = []
    file_name_array = []

    for i, filename in enumerate(train_val_set):
        savename = filename
        print(i, filename)

        graphics_img = cv2.imread(os.path.join(args.graphics_path, filename), -1)
        graphics_img = graphics_img[:, :, ::-1]
        graphics_img = np.array(graphics_img).astype(np.float32) / 255.0  # SYNTHIA images are 8 bit
        graphics_img = np.clip(graphics_img, 0, 1)

        # De-gamma
        graphics_img = graphics_img ** 2.2

        # Sample an illuminant
        while True:
            wb_vec = np.random.multivariate_normal(illum_mean, illum_cov, 1).squeeze()
            if in_hull(np.expand_dims(wb_vec[[0, 2]], axis=0), gt_illum[:, [0, 2]]):
                break
        gt_illum_array.append(wb_vec)
        print(wb_vec)

        # Compute CST matrix
        cst_mat = get_cst_matrix(cmD65, cmA, wb_vec)

        # sRGB to CIE XYZ (Invert XYZ2sRGB)
        raw_est = apply_combined_mat(graphics_img, np.linalg.inv(xyz2srgb_mat))

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
        cv2.imwrite(
            os.path.join(args.save_path, savename[:-4] + '.png'),
            (raw_est[:, :, [2, 1, 0]] * 65535).astype(np.uint16))
        file_name_array.append(savename[:-4])

    gt_values = {'gt_illum': gt_illum_array, 'filenames': file_name_array}

    pickle.dump(gt_values, open(os.path.join(args.save_path, 'gt_illum.p'), "wb"))

    print('Done')
