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

import time
from jobs.job_utils import prep_environment
from jobs.illum_est import train, test, ROOT_DIR
import argparse


if __name__ == '__main__':
    prep_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', '-n', default=5, type=int, help='Number of repeated runs.')
    args = parser.parse_args()

    method = 'upi'
    cameras = [
        'Canon1DsMkIII',
        'Canon600D',
        'FujifilmXM1',
        'NikonD5200',
        'OlympusEPL6',
        'PanasonicGX1',
        'SamsungNX2000',
        'SonyA57',
        'NikonD40'
    ]
    exp_id = 'illum_est'
    dataset_dir = f'{ROOT_DIR}/data'

    for i in range(args.num_runs):
        timestamp = int(time.time())
        exp_name = f'{exp_id}_{method}_{timestamp}'

        train('SamsungNX2000', method, exp_name, dataset_dir)

        for camera in cameras:
            test(camera, exp_name, dataset_dir)
            print(f'UPI testing done: {camera}')
