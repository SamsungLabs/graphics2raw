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
from data_generation.unprocess import unprocess
import tensorflow as tf


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
                        default='illum_est_expts/data/SamsungNX2000/upi'
                        )

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    RAND_SEED = 101
    np.random.seed(RAND_SEED)
    tf.random.set_seed(RAND_SEED)

    args = parse_args()

    train_val_set = pickle.load(open(args.train_val_set, 'rb'))

    gt_illum_array = []
    file_name_array = []

    for i, filename in enumerate(train_val_set):
        savename = filename
        print(i, filename)

        graphics_img = cv2.imread(os.path.join(args.graphics_path, filename), -1)
        graphics_img = graphics_img[:, :, ::-1]
        graphics_img = np.array(graphics_img).astype(np.float32) / 255.0  # SYNTHIA images are 8 bit
        graphics_img = np.clip(graphics_img, 0, 1)

        graphics_img = tf.convert_to_tensor(graphics_img, dtype=tf.float32)

        raw_est, metadata = unprocess(graphics_img)
        raw_est = raw_est.numpy().astype(np.float32)
        for k, v in metadata.items():
            metadata[k] = v.numpy().astype(np.float32)

        wb_vec = np.array([1 / metadata['red_gain'], 1, 1 / metadata['blue_gain']])

        gt_illum_array.append(wb_vec)
        print(wb_vec)

        cv2.imwrite(
            os.path.join(args.save_path, savename[:-4] + '.png'),
            (raw_est[:, :, [2, 1, 0]] * 65535).astype(np.uint16))
        file_name_array.append(savename[:-4])

    gt_values = {'gt_illum': gt_illum_array, 'filenames': file_name_array}

    pickle.dump(gt_values, open(os.path.join(args.save_path, 'gt_illum.p'), "wb"))

    print('Done')
