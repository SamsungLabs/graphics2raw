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

import glob
import cv2
import numpy as np
import pickle
import os
import scipy
from utils.img_utils import get_illum_normalized_by_g


def shuffle_files(fnames):
    """
    Shuffle graphics images to cover all scenes
    graphics_names: sorted graphics image names
    """
    np.random.seed(101)
    fnames = np.array(fnames)
    indices = np.arange(len(fnames))
    np.random.shuffle(indices)
    fnames = fnames[indices]
    return fnames


def get_all_patches(img, patch_size=48, stride=48):
    h, w, = img.shape[:2]
    patches = []
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[i:i + patch_size, j:j + patch_size, ...]
            patches.append(x)
    return patches


def gen_patches_illum_est(file_name, patch_size=48, stride=48):
    """
    Crop one image into patches
    :return:
    """
    img = np.array(cv2.imread(file_name, cv2.IMREAD_UNCHANGED))[:, :, ::-1].astype(np.float32)  # img is now in rgb
    img = img / 65535.0
    img = np.clip(img, 0, 1)
    patches = get_all_patches(img, patch_size, stride)
    return patches


def get_gt_illum_by_fname(illum_file):
    gt_illum_by_fname = {}
    if illum_file.endswith('.p'):
        content = pickle.load(open(illum_file, 'rb'))
        gt_illums = content['gt_illum']  # already normalized by g
        filenames = content['filenames']

    elif illum_file.endswith('.mat'):
        content = scipy.io.loadmat(illum_file)
        gt_illums = content['groundtruth_illuminants']
        gt_illums[:, 0], gt_illums[:, 1], gt_illums[:, 2] = get_illum_normalized_by_g(gt_illums)
        filenames = [name[0][0] + '_st4' for name in content['all_image_names']]  # array([array(['SamsungNX2000_0001'])) -> 'SamsungNX2000_0001_st4'
    else:
        raise Exception('Unsupported gt illum file type.')

    for illum, fname in zip(gt_illums, filenames):
        gt_illum_by_fname[fname] = illum
    return gt_illum_by_fname


def datagenerator_illum_est(dataset_dir='', illum_file='', split_file='', split='train',
                            batch_size=128, patch_size=48, stride=48, debug=False):
    file_list = sorted(glob.glob(os.path.join(dataset_dir, '*.png')))  # get name list of all .png files
    file_list = shuffle_files(file_list)  # shuffle to sample from all graphics scenes
    gt_illum_by_fname = get_gt_illum_by_fname(illum_file)

    split_indices = pickle.load(open(split_file, 'rb'))[split]
    file_list = file_list[split_indices]

    if debug:
        # For debugging only, check if images in the split are expected
        mysplit = 'valid' if split == 'val' else split
        split_fns_fp = split_file[:-5] + 'fns.p' 
        split_type = 'graphics_split' if dataset_dir.split('/')[-1] != 'real' else 'real_split'
        split_fns = pickle.load(open(split_fns_fp, 'rb'))[split_type][mysplit]
        for myf, gtf in zip(file_list, split_fns):
            assert os.path.basename(myf) == gtf, f'Panic! {os.path.basename(myf)}, {gtf}'
        print(f'Loaded images are correct.')
        # end of debugging

    in_patches = []
    gt_illums = []

    # generate patches
    for file in file_list:
        patches = gen_patches_illum_est(file, patch_size, stride)
        fname = os.path.basename(file)[:-4]
        gt_illum = gt_illum_by_fname[fname]
        in_patches.append(patches)
        gt_illums.append([gt_illum] * len(patches))

    in_patches = np.concatenate(in_patches)
    gt_illums = np.concatenate(gt_illums)
    discard_n = len(in_patches) - len(in_patches) // batch_size * batch_size
    in_patches = np.delete(in_patches, range(discard_n), axis=0)
    gt_illums = np.delete(gt_illums, range(discard_n), axis=0)

    print(f'^_^-{split} data finished-^_^')
    return in_patches, gt_illums

