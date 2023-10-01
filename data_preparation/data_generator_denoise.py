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

from pipeline.pipeline_utils import demosaic
from pipeline.pipeline import run_pipeline
from pipeline.raw_utils import stack_rggb_channels, RGGB2Bayer
from os.path import join
from noise_profiler.image_synthesizer import synthesize_noisy_image_v2


np.random.seed(101)


def get_all_patches(img, patch_size=48, stride=48):
    h, w, = img.shape[:2]
    patches = []
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[i:i + patch_size, j:j + patch_size, ...]
            patches.append(x)
    return patches


def normalize_s20fe(img):
    white_level = 1023.0
    black_level = 64.0
    img = (img - black_level) / (white_level - black_level)
    return img


def denormalize_s20fe(img):
    white_level = 1023.0
    black_level = 64.0
    img = img * (white_level - black_level) + black_level
    return img


def post_process_stacked_images_s20fe(img, cfa_pattern):
    """
    Process an image that went through stack_bayer_norm_gamma_s20fe
    to demosaiced, gamma-ed RAW RGB
    :param img:
    :param cfa_pattern:
    :return:
    """
    img = (img ** 2.2) * 1023.0  # [0, 1023]
    img = np.clip(img, 64, 1023)  # [64, 1023]
    img = (img - 64) / (1023 - 64)  # [0, 1] normalization with black level subtraction
    img = RGGB2Bayer(img)
    img = demosaic(img, cfa_pattern, alg_type='EA')
    img = img ** (1 / 2.2)
    return img


def post_process_stacked_images_s20fe_to_srgb(img, metadata):
    """
    Process an image that went through stack_bayer_norm_gamma_s20fe
    to sRGB
    :param img:
    :param metadata:
    :return:
    """
    img = (img ** 2.2) * 1023.0  # [0, 1023]
    img = np.clip(img, 64, 1023)  # [64, 1023]
    img = RGGB2Bayer(img)

    stages = [
        'raw',
        'normal',
        'lens_shading_correction',
        'white_balance',
        'demosaic',
        'xyz',
        'srgb',
        'fix_orient',
        'gamma',
    ]

    params = {
        'input_stage': 'raw',
        'output_stage': 'gamma',
        'demosaicer': 'EA',
    }

    img = run_pipeline(img, params=params, metadata=metadata, stages=stages)
    return img


def linearize_stacked_images_s20fe(img):
    """
    Process an image that went through stack_bayer_norm_gamma_s20fe
    to stacked, normalized, linear RAW RGB in the range of [0, 1]
    with black level subtraction
    :param img: a RAW image that went through stack_bayer_norm_gamma_s20fe
    :return:
    """
    img = img ** 2.2
    img = img * 1023.0  # [0, 1023], no black level subtraction
    img = normalize_s20fe(img)  # [0, 1], with black level subtraction
    return img


def stack_bayer_norm_gamma_s20fe(img, clip_bot=False):
    """

    :param img: not normalized bayer image, [black_level, white_level]
    :param clip_bot: clip minimum value; maximum value is always clipped to 1023
    :return:
    """
    img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1023.0)
    if clip_bot:
        img = np.clip(img, 64.0, 1023.0)
    img = img / 1023.0
    img = stack_rggb_channels(img)  # do not pass in bayer_pattern to not change the stacking order
    img = img ** (1 / 2.2)
    return img


def process_image(file_name, data_type='input', noise_model=None, iso=100):
    """
    Assumptions:
    Camera: S20FE
    h == 3024 and w == 4032

    :param iso: 1600 or 3200
    :param noise_model:  hg noise model
    :param data_type:
    input_graphics_raw: clean_raw -> noise -> stacked -> clip to 0~1023 -> gamma
    target_graphics_raw: clean_raw -> stacked -> clip to 64~1023 -> gamma
    input_real: noisy_raw -> stacked -> clip to 0~1023 -> gamma
    target_real: clean_raw -> stacked -> clip to 64~1023 -> gamma
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)  # bayer image
    h, w = img.shape[:2]
    assert h == 3024 and w == 4032, 'Wrong height and width!'

    # Generate input patches
    if data_type == 'input_graphics_raw':
        # Pre-generated noise
        img = synthesize_noisy_image_v2(img, model=noise_model['noise_model'],
                                        dst_iso=iso, min_val=0,
                                        max_val=1023,
                                        iso2b1_interp_splines=noise_model['iso2b1_interp_splines'],
                                        iso2b2_interp_splines=noise_model['iso2b2_interp_splines'])

        img_ = stack_bayer_norm_gamma_s20fe(img, clip_bot=False)

    # Generate target patches
    elif data_type == 'input_real':
        img_ = stack_bayer_norm_gamma_s20fe(img, clip_bot=False)
    else:
        # Covered cases: target_graphics_raw, target_real
        # img: linear, [black_level, white_level], bayer
        img_ = stack_bayer_norm_gamma_s20fe(img, clip_bot=True)

    img_ = img_.astype(np.float32)
    return img_


def datagenerator_raw(data_dir='dummy_dataset/train', batch_size=128, patch_size=48, stride=48, verbose=True,
                      data_type='graphics_raw', noise_model=None, iso=3200):
    if 'graphics' in data_type:
        input_list = sorted(glob.glob(join(data_dir, 'clean_raw', '*.png')))
        target_list = input_list
    elif 'real' in data_type:
        input_list = sorted(glob.glob(join(data_dir, 'noisy_raw', '*.png')))
        target_list = sorted(glob.glob(join(data_dir, 'clean_raw', '*.png')))
    else:
        raise Exception('Unexpected data type')

    # initialize
    patches_input = []
    patches_target = []
    # generate patches
    for i in range(len(input_list)):
        img_input = process_image(input_list[i], data_type=f'input_{data_type}', noise_model=noise_model, iso=iso)
        img_target = process_image(target_list[i], data_type=f'target_{data_type}')
        img_pair = np.dstack([img_input, img_target])
        patches = get_all_patches(img_pair, patch_size, stride)
        patches = np.array(patches)
        assert patches[0].shape[-1] == 8  # both input and target are stacked
        split = 4
        patch_input = patches[..., :split]
        patch_target = patches[..., split:]
        patches_input.append(patch_input)
        patches_target.append(patch_target)
        if verbose:
            print(str(i + 1) + '/' + str(len(input_list)) + ' is done')

    patches_input = np.concatenate(patches_input)
    patches_target = np.concatenate(patches_target)

    discard_n = len(patches_input) - len(patches_input) // batch_size * batch_size

    patches_input = np.delete(patches_input, range(discard_n), axis=0)
    patches_target = np.delete(patches_target, range(discard_n), axis=0)

    print('^_^-raw training data finished-^_^')
    return patches_input, patches_target

