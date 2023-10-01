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
from jobs.neural_isp_graphics2raw import movedata, train, test
import argparse


if __name__ == '__main__':
    prep_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-t', default=None, type=str, help='Time stamp to distinguish between runs.')
    parser.add_argument('--eval', '-e', action='store_true', help='Evaluation only, no training.')
    args = parser.parse_args()
    timestamp = args.timestamp if args.timestamp else int(time.time())

    input_type = 'clean_raw'
    exp_name = 'neural_isp_upi'

    input_dir = 'graphics_dngs_upi'
    target_dir = 'graphics_srgb_upi'
    movedata(input_dir, target_dir)
    if not args.eval:
        train(input_type, exp_name, timestamp)
    test(input_type, exp_name, timestamp)




