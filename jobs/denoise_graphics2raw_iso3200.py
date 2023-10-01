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
import shutil
from jobs.job_utils import prep_environment, get_day2night_data

DATA_ROOT_DIR = 'neural_isp_expts'  # uses the same RAW data as neural ISP
EXPT_ROOT_DIR = 'denoising_expts'


def movedata(input_dir):
    print('Preparing data...')
    get_day2night_data()
    os.system(f'python3 -m data_preparation.initial_data_prep_denoise --dng_dir {DATA_ROOT_DIR}/data/{input_dir};')


def train(exp_name, timestamp, iso):
    folder_name = f'{exp_name}_{timestamp}'
    args_str = f"--savefoldername {folder_name} \
            --exp_dir {EXPT_ROOT_DIR}/expts/ \
            --lr 0.0001 \
            --num_epochs 100 \
            --milestones 90 \
            --patch_size 128 \
            --batch_size 32 \
            --loss l1 \
            --restormer_dim 8 \
            --data_type graphics_raw \
            --iso {iso} \
            --model_save_freq 400"  # not saving intermediate models

    print('Start training...')
    print('args_str = {}'.format(args_str))
    os.system('python3 train_denoise.py {} '.format(args_str))


def test(exp_name, timestamp, iso):
    folder_name = f'{exp_name}_{timestamp}'
    print('Start testing...')
    args_str = f"--model_dir {folder_name} \
            --exp_dir {EXPT_ROOT_DIR}/expts/ \
            --set_name iso_{iso} \
            --restormer_dim 8"

    print('Start testing...')
    print('args_str = {}'.format(args_str))
    os.system('python3 test_denoise.py {} '.format(args_str))

    target_dir = f'{EXPT_ROOT_DIR}/expts/{folder_name}/results/'
    os.makedirs(target_dir, exist_ok=True)
    os.system(f'cp results/* {target_dir} -r -v;')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', '-n', default=1, type=int, help='Number of runs.')
    return parser


if __name__ == '__main__':
    prep_environment()
    parser = get_parser()
    args = parser.parse_args()
    
    iso = 3200
    exp_name = f'denoise_graphics2raw_iso{iso}'

    input_dir = 'graphics_dngs_graphics2raw'
    movedata(input_dir)

    num_runs = args.num_runs
    for i in range(num_runs):
        timestamp = int(time.time())
        train(exp_name, timestamp, iso)
        test(exp_name, timestamp, iso)
        print(f'-----------Experiment: {exp_name}_{timestamp}-----------')

        shutil.rmtree('results')
