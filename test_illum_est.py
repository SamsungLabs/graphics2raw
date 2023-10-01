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

from model_archs.cnn import IllumEstNet
from utils.general_utils import save_args
import argparse
import os, time, datetime
import numpy as np
import cv2
import torch
from utils.general_utils import check_dir
from data_preparation.data_generator_illum_est import shuffle_files, get_gt_illum_by_fname
import glob
import pickle
from utils.img_utils import compute_ang_error


def to_tensor(img):
    img = torch.from_numpy(img.astype(np.float32))
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    return img


def from_tensor(img):
    img = img.permute(0, 2, 3, 1)
    img = img.cpu().detach().numpy()
    return np.squeeze(img)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_results(ang_err_array, out_fn=None, method=''):
    ang_err_array = np.array(ang_err_array)
    err_arr_sorted = np.sort(ang_err_array)
    fourth = int(np.round(ang_err_array.shape[0] / 4.0))
    best25_mean_err = np.round(np.mean(err_arr_sorted[:fourth]), 4)
    worst25_mean_err = np.round(np.mean(err_arr_sorted[-fourth:]), 4)
    arr_len = ang_err_array.shape[0]
    mean_err = np.round(ang_err_array.mean(), 4)
    median_err = np.round(np.median(ang_err_array), 4)

    print(arr_len, mean_err, median_err, best25_mean_err, worst25_mean_err)

    if out_fn:
        if not os.path.exists(out_fn):
            f = open(out_fn, 'w')
            f.write('exp / angular_errs, num_test, mean, median, best 25%, worst 25%\n')
        else:
            f = open(out_fn, 'a')
        f.write(f'{method}, {arr_len}, {mean_err}, {median_err}, {best25_mean_err}, {worst25_mean_err}\n')
        f.close()


if __name__ == '__main__':

    """
    Assumed dataset directory structure:
    dataset_root_dir
        camera
            method

    Assumed experiment directory structure:
    exp_dir
        exp_name
            models
            results
                bestmodel
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', default='illum_est_expts/data/ours', type=str, help='folder of png images')
    parser.add_argument(
        '--illum_file', default='illum_est_expts/data/ours/gt_illum.p', type=str,
        help='path to split indices')
    parser.add_argument(
        '--split_file', default='illum_est_expts/data/SamsungNX2000_train_valid_test_split_idx.p', type=str,
        help='path to split indices')
    parser.add_argument('--exp_name', default='illum_est_expt', type=str, help='experiment name, relative to exp_dir')
    parser.add_argument('--model_name', default='bestmodel.pt', type=str, help='name of the model')
    parser.add_argument('--num_filters', type=int, default=7, help='number of filters for CNN layers ')
    parser.add_argument('--exp_dir', default='./', type=str, help='directory to save experiment data to')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = args.exp_name

    model = IllumEstNet(in_channels=3, out_channels=3, num_filters=args.num_filters)

    model.load_state_dict(torch.load(os.path.join(args.exp_dir, exp_name, 'models', args.model_name)))
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    print('model loaded')

    full_save_path = os.path.join(args.exp_dir, exp_name, 'results', args.model_name[:-3])
    check_dir(full_save_path)
    save_args(args, full_save_path)

    ang_errs = []
    ssims = []

    camera, method = args.dataset_dir.split('/')[-2:]  # assume dataset_dir points to .../camera/method
    fname_result = f'results_{camera}.txt'
    f_result = open(os.path.join(full_save_path, fname_result), "w")

    # Data loading
    file_list = sorted(glob.glob(os.path.join(args.dataset_dir, '*.png')))  # get name list of all .png files
    file_list = shuffle_files(file_list)  # shuffle to sample from all scenes
    gt_illum_by_fname = get_gt_illum_by_fname(args.illum_file)

    split_indices = pickle.load(open(args.split_file, 'rb'))['test']
    file_list = file_list[split_indices]

    if args.debug:
        # For debugging only, check if images in the split are expected
        split_fns_fp = args.split_file[:-5] + 'fns.p'
        split_type = 'graphics_split' if args.dataset_dir.split('/')[-1] != 'real' else 'real_split'
        split_fns = pickle.load(open(split_fns_fp, 'rb'))[split_type]['test']
        for myf, gtf in zip(file_list, split_fns):
            assert os.path.basename(myf) == gtf, f'Panic! {os.path.basename(myf)}, {gtf}'
        print(f'Loaded images are correct.')
        # end of debugging

    for file in file_list:
        img = np.array(cv2.imread(file, cv2.IMREAD_UNCHANGED))[:, :, ::-1].astype(np.float32)  # img is now rgb
        img = img / 65535.0
        img = np.clip(img, 0, 1)

        fname = os.path.basename(file)[:-4]
        gt_illum = gt_illum_by_fname[fname]

        x = to_tensor(img)
        x = x.to(device)
        start_time = time.time()
        with torch.no_grad():
            y_ = model(x)  # inference
        elapsed_time = time.time() - start_time
        y_ = y_.cpu().numpy().squeeze()
        ang_err = compute_ang_error(y_, gt_illum)
        ang_errs.append(ang_err)

        log('{0:10s} \n ang_err = {1:2.2f} deg, Time = {2:2.4f} seconds, Pred: {3}, GT: {4}'.format(fname, ang_err,
                                                                                                    elapsed_time,
                                                                                                    y_, gt_illum))
        f_result.write('{0:10s} : ang_err = {1:2.2f} deg, '
                       'Time = {2:2.4f} seconds, Pred: {3}, GT: {4} \n'.format(fname,
                                                                               ang_err, elapsed_time, y_, gt_illum))

    ang_err_avg = np.mean(ang_errs)

    print()
    log('Dataset: {0:10s} \n Avg. Ang Err = {1:2.4f} deg'.format(f'{camera}_{method}', ang_err_avg))

    f_result.write('\nDataset: {0:10s} \n Avg. Ang Err = {1:2.4f} deg'.format(f'{camera}_{method}', ang_err_avg))
    f_result.close()

    aggr_result_fp = os.path.join(args.exp_dir, 'results.csv')
    index = exp_name.find(method)
    result_header = f'{camera}_{exp_name}' if camera not in exp_name else exp_name
    save_results(np.array(ang_errs), aggr_result_fp, method=result_header)
