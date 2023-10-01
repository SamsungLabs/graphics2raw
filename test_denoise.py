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

from model_archs.restormer import Restormer
from utils.general_utils import save_args, get_git_info, str2int_arr
import argparse
import os, time, datetime
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2
import torch
from pipeline.pipeline_utils import get_metadata, get_visible_raw_image
from utils.general_utils import check_dir
from data_preparation.data_generator_denoise import post_process_stacked_images_s20fe, stack_bayer_norm_gamma_s20fe, \
    linearize_stacked_images_s20fe, post_process_stacked_images_s20fe_to_srgb
import glob


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


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        cv2.imwrite(path, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='real_dataset', type=str, help='directory of test dataset')
    parser.add_argument('--set_name', default='iso_50', type=str, help='name of test dataset')
    parser.add_argument('--model_dir', default='no_wb_model', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='bestmodel.pt', type=str, help='name of the model')
    parser.add_argument('--restormer_dim', default=8, type=int, help='Restormer dim.')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_num', default=0, type=int, help='number of images to save')
    parser.add_argument('--save_visual', action='store_true', help='save save_num number of images')
    parser.add_argument('--save_visual_only', action='store_true', help='only save visual results')
    parser.add_argument('--save_fns', default='', type=str,
                        help='path to a file specifying which images to generate visual results for. '
                             'Used only when save_visual_only is True.')

    parser.add_argument('--exp_dir', default='./', type=str, help='directory to save experiment data to')
    args = parser.parse_args()

    to_save = []
    if args.save_visual_only:
        with open(args.save_fns, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        to_save = lines

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Restormer(inp_channels=4,
                      out_channels=4,
                      dim=args.restormer_dim,
                      num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4,
                      heads=[1, 2, 4, 8],
                      ffn_expansion_factor=2.66,
                      bias=False,
                      LayerNorm_type='BiasFree',
                      dual_pixel_task=False)

    model.load_state_dict(torch.load(os.path.join(args.exp_dir, args.model_dir, 'models', args.model_name)))
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    print('model loaded')

    ### File IO
    set_name = args.set_name
    fullsavepath = os.path.join(args.result_dir,
                                os.path.basename(args.set_dir) + '_' + set_name + '_' + os.path.basename(
                                    args.model_dir) + '_' + args.model_name[:-3])
    check_dir(args.result_dir)
    check_dir(fullsavepath)
    save_args(args, fullsavepath)

    with open(os.path.join(fullsavepath, 'test_info.txt'), 'w') as f:
        f.write(get_git_info())

    psnrs = []
    ssims = []

    if not args.save_visual_only:
        fsrgb = open(os.path.join(fullsavepath, "results.txt"), "w")

    ### Inference
    test_dir = args.set_dir
    for c, dng_path in enumerate(sorted(glob.glob(os.path.join(args.set_dir, 'dng', set_name, '*.dng')))):
        im = os.path.basename(dng_path)
        if args.save_visual_only:
            tag = im[:-4]
            if tag not in to_save:
                print('Skipping ', tag)
                continue

        print('Processing ' + str(c + 1))
        if set_name == 'iso_50':
            x = cv2.imread(os.path.join(args.set_dir, 'clean_raw', im[:-4] + '.png'), cv2.IMREAD_UNCHANGED)
        else:
            x = get_visible_raw_image(os.path.join(args.set_dir, 'dng', set_name, im))

        meta_path = os.path.join(args.set_dir, 'dng', set_name, im)
        meta_data = get_metadata(meta_path)

        x_gamma = stack_bayer_norm_gamma_s20fe(x, clip_bot=False)
        x_gamma = to_tensor(x_gamma)

        y = np.array(cv2.imread(os.path.join(args.set_dir, 'clean_raw', im[:-4] + '.png'), cv2.IMREAD_UNCHANGED))
        y_gamma = stack_bayer_norm_gamma_s20fe(y, clip_bot=True)
        y_gamma = np.clip(y_gamma, 0, 1)

        x_gamma = x_gamma.to(device)
        start_time = time.time()
        with torch.no_grad():
            pred_gamma = model(x_gamma)  # inference
        elapsed_time = time.time() - start_time
        pred_gamma = pred_gamma.cpu()

        pred_gamma = from_tensor(pred_gamma)
        pred_gamma = np.clip(np.squeeze(pred_gamma), 0, 1)

        y_lin = linearize_stacked_images_s20fe(y_gamma)
        pred_lin = linearize_stacked_images_s20fe(pred_gamma)
        psnr_x = compare_psnr(y_lin, pred_lin, data_range=1)
        ssim_x = compare_ssim(y_lin, pred_lin, multichannel=True, data_range=1)
        psnrs.append(psnr_x)
        ssims.append(ssim_x)

        log('{0:10s} \n PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, Time = {3:2.4f} seconds'.format(im, psnr_x,
                                                                                            ssim_x, elapsed_time))
        if not args.save_visual_only:
            fsrgb.write('{0:10s} : PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, Time = {3:2.4f} seconds \n'.format(im, psnr_x,
                                                                                                          ssim_x,
                                                                                                          elapsed_time))

        if (args.save_visual and c < args.save_num) or args.save_visual_only:
            x_gamma = from_tensor(x_gamma).squeeze()  # drop the batch dimension, assume bs = 1
            # 4-channel images
            y_gamma_ = (255 * post_process_stacked_images_s20fe(y_gamma, meta_data['cfa_pattern'])).astype('uint8')
            pred_gamma_ = (255 * post_process_stacked_images_s20fe(pred_gamma, meta_data['cfa_pattern'])).astype('uint8')
            x_gamma_ = (255 * post_process_stacked_images_s20fe(x_gamma, meta_data['cfa_pattern'])).astype('uint8')

            name, ext = os.path.splitext(im)
            save_result(x_gamma_[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + '_input.png')) 
            save_result(pred_gamma_[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + f'_output_psnr{np.round(psnr_x, 4)}_ssim{np.round(ssim_x, 4)}.png')) 
            save_result(y_gamma_[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + '_target.png')) 
            
            y_srgb = (255 * post_process_stacked_images_s20fe_to_srgb(y_gamma, meta_data)).astype('uint8')
            pred_srgb = (255 * post_process_stacked_images_s20fe_to_srgb(pred_gamma, meta_data)).astype('uint8')
            x_srgb = (255 * post_process_stacked_images_s20fe_to_srgb(x_gamma, meta_data)).astype('uint8')

            save_result(x_srgb[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + '_input_srgb.png')) 
            save_result(pred_srgb[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + f'_output_srgb_psnr{np.round(psnr_x, 4)}_ssim{np.round(ssim_x, 4)}.png')) 
            save_result(y_srgb[:, :, [2, 1, 0]],
                        path=os.path.join(fullsavepath, name + '_target_srgb.png'))  

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    psnrs.append(psnr_avg)
    ssims.append(ssim_avg)

    print()
    log('Dataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}'.format(set_name, psnr_avg, ssim_avg))

    if not args.save_visual_only:
        fsrgb.write(
            '\nDataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}'.format(set_name, psnr_avg, ssim_avg))
        fsrgb.close()
