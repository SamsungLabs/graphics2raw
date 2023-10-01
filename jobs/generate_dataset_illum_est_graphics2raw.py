"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
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
from jobs.job_utils import prep_environment
"""
Requirements:
illum_est_expts/data/<camera>/real
illum_est_expts/data/<camera>/real/<camera>_gt.mat
illum_est_expts/nus_metadata/
illum_est_expts/nus_metadata/nus_outdoor_gt_illum_mats
illum_est_expts/synthia/SYNTHIA_RAND_CVPR16/RGB
"""


def main():
    graphics_path = 'illum_est_expts/synthia/SYNTHIA_RAND_CVPR16/RGB'
    
    cameras = [
        'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200', 'OlympusEPL6', 
        'PanasonicGX1',
        'SamsungNX2000', 
        'SonyA57',
        'NikonD40'
    ]

    method = 'ours'

    for camera in cameras:
        camera_dir = os.path.join('illum_est_expts/data', camera)
        save_path = os.path.join(camera_dir, method)

        os.makedirs(save_path, exist_ok=True)

        mat_file = os.path.join('illum_est_expts/nus_metadata', 'nus_outdoor_gt_illum_mats',
                                f'{camera}_gt.mat')
        split_file = f'assets/split_files/illum_est/{camera}_train_valid_test_split_fns.p'

        assert os.path.exists(mat_file)
        assert os.path.exists(split_file)
        
        args_str = f"--save_path {save_path} \
                --target_camera {camera} \
                --mat_file {mat_file} \
                --use_train_val_illums_only \
                --split_file {split_file} \
                --graphics_path {graphics_path} \
                --rgb_gain \
                --rgb_gain_mean 0.8"
                # do safe invert
                # max illums None

        os.system('python3 -m data_generation.invert_synthia_graphics2raw {} '.format(args_str))


if __name__ == '__main__':
    prep_environment()
    main()





