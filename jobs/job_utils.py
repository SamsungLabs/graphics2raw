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


def prep_environment():
    os.system('pip install -r requirements.txt')


def get_day2night_data():
    # copy subdirectories in night_real into ./real_dataset 
    assert os.path.exists('real_dataset')
    assert os.path.exists('real_dataset/clean')
    assert os.path.exists('real_dataset/clean_raw')
    assert os.path.exists('real_dataset/dng')
