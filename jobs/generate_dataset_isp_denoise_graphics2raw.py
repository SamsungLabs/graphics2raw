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


def main():
    save_path = 'neural_isp_expts/data/graphics_dngs_graphics2raw'
    os.makedirs(save_path, exist_ok=True)

    exr_folder_path = 'path/to/graphics2RAW-dataset/'

    args_str = f"--save_path {save_path} \
            --exr_folder_path {exr_folder_path} \
            --rgb_gain \
            --rgb_gain_mean 0.8"
    # do apply safe invert

    os.system('python3 -m data_generation.package_exr_to_dng_graphics2raw {} '.format(args_str))


if __name__ == '__main__':
    prep_environment()
    main()





