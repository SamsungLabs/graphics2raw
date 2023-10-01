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
from jobs.denoise_real_iso3200 import get_parser, movedata, train, test
from jobs.job_utils import prep_environment
import shutil


if __name__ == '__main__':
    prep_environment()
    parser = get_parser()
    args = parser.parse_args()

    fold = args.fold
    iso = 1600
    exp_name = f'denoise_real_iso{iso}_f{fold}'

    movedata(fold)

    num_runs = args.num_runs
    for i in range(num_runs):
        timestamp = int(time.time())
        train(exp_name, timestamp, args.restormer_dim, args.epochs)
        test(exp_name, timestamp, args.restormer_dim)
        print(f'-----------Experiment: {exp_name}_{timestamp}-----------')

        shutil.rmtree('results')
