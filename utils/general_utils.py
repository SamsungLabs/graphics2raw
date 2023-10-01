"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Abhijith Punnappurath (abhijith.p@samsung.com)
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
import errno
import subprocess
import sys


def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def save_args(args, save_dir):
    """
    Source: https://github.com/VITA-Group/EnlightenGAN/blob/master/options/base_options.py
    EnlightenGAN base_options.py
    """
    args = vars(args)
    file_name = os.path.join(save_dir, 'args.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

        opt_file.write('\n------------------------------\n')
        opt_file.write('Shell command:\n')
        opt_file.write(get_command())


def get_git_revision_hash() -> str:
    """
    Source: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    :return:
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_branch() -> str:
    return subprocess.check_output(['git', 'branch']).decode('ascii').strip()


def get_git_info() -> str:
    current_hash = get_git_revision_hash()
    current_branch = get_git_revision_branch()
    git_info = f'Git Info:\nCurrent commit: {current_hash}\nBranches:\n {current_branch}'
    return git_info


def str2int_arr(arr):
    # Parse comma-splited integer array
    return [int(e) for e in arr.split(',')]


def get_command() -> str:
    return " ".join(sys.argv[:])
