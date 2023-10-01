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
from jobs.job_utils import prep_environment
import argparse


ROOT_DIR = 'illum_est_expts'

def train(camera, method, exp_name, dataset_dir):
    illum_fname = f'{camera}_gt.mat' if method == 'real' else 'gt_illum.p'

    args_str = f"--dataset-dir {os.path.join(dataset_dir, camera, method)} \
            --savefoldername {exp_name} \
            --illum_file {os.path.join(dataset_dir, camera, method, illum_fname)} \
            --split_file assets/split_files/illum_est/{camera}_train_valid_test_split_idx.p \
            --exp_dir {ROOT_DIR}/expts \
            --on-cuda \
            --model_save_freq 3000 \
            --num-epochs 2000"
            # not saving intermediate models

    print('Start training...')
    print('args_str = {}'.format(args_str))
    os.system('python3 train_illum_est.py {} '.format(args_str))


def test(camera, exp_name, dataset_dir):
    print('Start testing...')
    args_str = f"--dataset-dir {os.path.join(dataset_dir, camera, 'real')} \
            --exp_name {exp_name} \
            --illum_file {os.path.join(dataset_dir, camera, 'real', f'{camera}_gt.mat')} \
            --split_file assets/split_files/illum_est/{camera}_train_valid_test_split_idx.p \
            --exp_dir {ROOT_DIR}/expts"

    print('Start testing...')
    print('args_str = {}'.format(args_str))
    os.system('python3 test_illum_est.py {} '.format(args_str))


if __name__ == '__main__':
    prep_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras', '-c', default='SamsungNX2000,SonyA57', type=str, help='Which cameras to run on. Comma-separated string.')
    parser.add_argument('--methods', '-m', default='ours,real', type=str, help='Which methods to run on. Comma-separated string.')
    parser.add_argument('--num_runs', '-n', default=5, type=int, help='Number of repeated runs.')
    args = parser.parse_args()

    exp_id = 'illum_est'
    cameras = [] if args.cameras == '' else args.cameras.split(',')
    methods = [] if args.methods == '' else args.methods.split(',')

    for method in methods:
        for camera in cameras:
            for i in range(args.num_runs):
                timestamp = int(time.time())
                exp_name = f'{camera}_{exp_id}_{method}_{timestamp}'

                dataset_dir = f'{ROOT_DIR}/data'
                train(camera, method, exp_name, dataset_dir)
                test(camera, exp_name, dataset_dir)
