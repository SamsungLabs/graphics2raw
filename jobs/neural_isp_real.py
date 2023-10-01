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
from jobs.neural_isp_graphics2raw import ROOT_DIR


def movedata(fold):
    print('Preparing data...')
    get_day2night_data()
    
    tif_dir = f'{ROOT_DIR}/data/clean_srgb_tiffs'
    os.system(f'python3 -m data_preparation.initial_data_prep_tif2png --tif_dir {tif_dir};')
    os.system(f'python3 -m data_preparation.k_fold_split_data --which_fold {fold} --with_noise 0;')


def train(input_type, folder_name):
    args_str = f"--which-input {input_type} \
            --savefoldername {folder_name} \
            --exp_dir {ROOT_DIR}/expts/ \
            --on-cuda \
            --data-dir real_dataset_k_fold \
            --model_save_freq 600 \
            --num-epochs 250"

    print('Start training...')
    print('args_str = {}'.format(args_str))
    os.system('python3 train_neural_isp.py {} '.format(args_str))


def test(folder_name):
    args_str = f"--exp_dir {ROOT_DIR}/expts/ \
            --model_dir {folder_name} \
            --set_dir real_dataset_k_fold/test"

    print('Start testing...')
    print('args_str = {}'.format(args_str))
    os.system('python3 test_neural_isp.py {} '.format(args_str))

    target_dir = f'{ROOT_DIR}/expts/{folder_name}/results/'
    os.makedirs(target_dir, exist_ok=True)
    os.system(f'cp results/* {target_dir} -r -v;')


if __name__ == '__main__':
    prep_environment()

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', '-f', default=0, type=int, help='Which fold of the real data to train and test.')
    args = parser.parse_args()
    fold = args.fold

    input_type = 'clean_raw'
    exp_name = 'neural_isp_real'
    num_runs = 2
    for i in range(num_runs):
        timestamp = int(time.time())
        folder_name = f'{exp_name}_{input_type}_{fold}_{timestamp}'
        movedata(fold)
        train(input_type, folder_name)
        test(folder_name)
        print(f'-----------Experiment: {folder_name}-----------')

        shutil.rmtree('real_dataset')
        shutil.rmtree('real_dataset_k_fold')
        shutil.rmtree('results')
