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
import shutil
from jobs.denoise_graphics2raw_iso3200 import train, test, get_parser, prep_environment, movedata


if __name__ == '__main__':
    prep_environment()
    parser = get_parser()
    args = parser.parse_args()

    iso = 3200
    exp_name = f'denoise_upi_iso{iso}'
    input_dir = 'graphics_dngs_upi'
    movedata(input_dir)

    num_runs = args.num_runs
    for i in range(num_runs):
        timestamp = int(time.time())
        train(exp_name, timestamp, iso)
        test(exp_name, timestamp, iso)
        print(f'-----------Experiment: {exp_name}_{timestamp}-----------')

        shutil.rmtree('results')
