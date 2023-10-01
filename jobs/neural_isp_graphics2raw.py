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
import time
import argparse
from jobs.job_utils import prep_environment, get_day2night_data


ROOT_DIR = 'neural_isp_expts'

def movedata(input_dir, target_dir):
    print('Preparing data...')
    get_day2night_data()
    
    tif_dir = f'{ROOT_DIR}/data/clean_srgb_tiffs'
    os.system(f'python3 -m data_preparation.initial_data_prep_tif2png --tif_dir {tif_dir};')
    os.system(f'python3 -m data_preparation.initial_data_prep_neural_isp --dng_dir {ROOT_DIR}/data/{input_dir} --tif_dir {ROOT_DIR}/data/{target_dir};')


def train(input_type, exp_name, timestamp):
    args_str = f"--which-input {input_type} \
            --savefoldername {exp_name}_{input_type}_{timestamp} \
            --exp_dir {ROOT_DIR}/expts/ \
            --on-cuda \
            --model_save_freq 600 \
            --num-epochs 250"

    print('Start training...')
    print('args_str = {}'.format(args_str))
    os.system('python3 train_neural_isp.py {} '.format(args_str))


def test(input_type, exp_name, timestamp):
    print('Start testing...')
    folder_name = f'{exp_name}_{input_type}_{timestamp}'
    args_str = f"--exp_dir {ROOT_DIR}/expts/ \
                 --model_dir {folder_name}"

    print('Start testing...')
    print('args_str = {}'.format(args_str))
    os.system('python3 test_neural_isp.py {} '.format(args_str))

    target_dir = f'{ROOT_DIR}/expts/{folder_name}/results/'
    os.makedirs(target_dir, exist_ok=True)
    os.system(f'cp results/* {target_dir} -r -v;')


if __name__ == '__main__':
    prep_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-t', default=None, type=str, help='Time stamp to distinguish between runs.')
    parser.add_argument('--eval', '-e', action='store_true', help='Evaluation only, no training.')
    args = parser.parse_args()
    timestamp = args.timestamp if args.timestamp else int(time.time())

    input_type = 'clean_raw'
    exp_name = 'neural_isp_graphics2raw'

    input_dir = 'graphics_dngs_graphics2raw'
    target_dir = 'graphics_srgb_graphics2raw'
    movedata(input_dir, target_dir)
    if not args.eval:
        train(input_type, exp_name, timestamp)
    test(input_type, exp_name, timestamp)




