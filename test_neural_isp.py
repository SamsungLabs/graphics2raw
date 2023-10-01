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

from model_archs.unet import UNet
from utils.general_utils import save_args
import argparse
import os, time, datetime
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2
import torch
from pipeline.pipeline import run_pipeline
from pipeline.pipeline_utils import get_metadata, get_visible_raw_image
from utils.general_utils import check_dir
from utils.img_utils import calc_deltaE_rgb


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
    parser.add_argument('--model_dir', default='neural_isp_model', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='bestmodel.pt', type=str, help='name of the model')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', action='store_true', help='save image')
    parser.add_argument('--save_num', default=0, type=int, help='number of images to save')
    parser.add_argument('--is-PS-sRGB', default=True, action='store_true', help='False: our in-house ISP sRGB like in day-to-night experiments, True: Photoshop sRGB')
    parser.add_argument('--num_filters', type=int, default=32, help='number of filters for UNet layers ')
    parser.add_argument('--exp_dir', default='./', type=str, help='directory to save experiment data to')
    parser.add_argument('--save_visual_only', action='store_true', help='True: only save visual results; save_result '
                                                                        'and save_num will not be used.')
    parser.add_argument('--save_fns', default='', type=str,
                        help='path to a file specifying which images to generate visual results for. '
                             'Used only when save_visual_only is True.')
    args = parser.parse_args()

    to_save = []
    if args.save_visual_only:
        with open(args.save_fns, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        to_save = lines

    if args.is_PS_sRGB:
        params = {
            'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
            'white_balancer': 'default',  # options: default, or self-defined module
            'demosaicer': '',  # options: '' for simple interpolation,
            #          'EA' for edge-aware,
            #          'VNG' for variable number of gradients,
            #          'menon2007' for Menon's algorithm
            'tone_curve': 'simple-s-curve',  # options: 'simple-s-curve', 'default', or self-defined module
            'output_stage': 'default_cropping',
        }

        stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'lens_shading_correction',
                'white_balance', 'demosaic', 'default_cropping']
    else:
        params = {
            'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
            'white_balancer': 'default',  # options: default, or self-defined module
            'demosaicer': '',  # options: '' for simple interpolation,
            #          'EA' for edge-aware,
            #          'VNG' for variable number of gradients,
            #          'menon2007' for Menon's algorithm
            'tone_curve': 'simple-s-curve',  # options: 'simple-s-curve', 'default', or self-defined module
            'output_stage': 'demosaic',
        }

        stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
                  'demosaic']

    set_names_list = ['iso_50']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3, init_features=args.num_filters)
    
    model_path = os.path.join(args.exp_dir, args.model_dir, 'models', args.model_name)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    print('model loaded')

    for folder_index, set_cur in enumerate(set_names_list):
        print(set_cur)

        fullsavepath = os.path.join(args.result_dir,
                                   os.path.basename(args.set_dir) + '_' + set_cur + '_' + os.path.basename(args.model_dir) + '_' + args.model_name[:-3])

        check_dir(args.result_dir)
        check_dir(fullsavepath)
        save_args(args, fullsavepath)

        psnrs = []
        ssims = []
        deltaEs = []

        if not args.save_visual_only:
            fsrgb = open(os.path.join(fullsavepath, "results.txt"), "w")

        for c, im in enumerate(sorted(os.listdir(os.path.join(args.set_dir, 'dng', set_cur)))):
            if args.save_visual_only:
                tag = im[:-4]
                if tag not in to_save:
                    continue

            print('Processing ' + str(c + 1) + ' ' + os.path.basename(im))
            if set_cur == 'iso_50':
                x = cv2.imread(os.path.join(args.set_dir, 'clean_raw', im[:-4]+'.png'), cv2.IMREAD_UNCHANGED)
            else:
                x = get_visible_raw_image(os.path.join(args.set_dir, 'dng', set_cur, im))

            meta_path = os.path.join(args.set_dir, 'dng', set_cur, im)
            meta_data_org = get_metadata(meta_path)

            x = run_pipeline(x, params=params, metadata=meta_data_org, stages=stages)

            x = x ** (1 / 2.2)

            x = to_tensor(x)

            y = cv2.imread(os.path.join(args.set_dir, 'clean', im[:-4]+'.png'), cv2.IMREAD_UNCHANGED)
            y = np.array(cv2.cvtColor(y, cv2.COLOR_BGR2RGB), dtype=np.float32)
            if y.shape[0] > y.shape[1]:
                y = np.rot90(y)
            y = y.astype('uint8')

            x = x.to(device)
            start_time = time.time()
            with torch.no_grad():
                y_ = model(x)  # inference
            elapsed_time = time.time() - start_time
            y_ = y_.cpu()

            y_ = (y_).permute(0, 2, 3, 1).numpy()
            y_ = np.clip(np.squeeze(y_), 0, 1)
            y_ = (255 * y_).astype('uint8')

            psnr_x_ = compare_psnr(y, y_, data_range=255)
            ssim_x_ = compare_ssim(y, y_, multichannel=True, data_range=255)
            deltaE_x_ = calc_deltaE_rgb(y, y_)
            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)
            deltaEs.append(deltaE_x_)

            if (args.save_result and c < args.save_num) or args.save_visual_only:
                name, ext = os.path.splitext(im)
                x_ = x.permute(0, 2, 3, 1).cpu().numpy().squeeze()  # drop the batch dimension, assume bs = 1
                x_ = (255 * x_).astype('uint8')
                save_result(x_[:, :, [2, 1, 0]],
                            path=os.path.join(fullsavepath, name + '_input.png'))
                save_result(y_[:, :, [2, 1, 0]],
                            path=os.path.join(fullsavepath, name + f'_output_psnr{np.round(psnr_x_, 4)}_ssim{np.round(ssim_x_, 4)}.png'))
                save_result(y[:, :, [2, 1, 0]],
                            path=os.path.join(fullsavepath, name + '_target.png'))

            log('{0:10s} \n PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, DeltaE = {3:2.4f} Time = {4:2.4f} seconds'.format(im, psnr_x_,
                                                                                            ssim_x_, deltaE_x_, elapsed_time))
            if not args.save_visual_only:
                fsrgb.write('{0:10s} : PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, DeltaE = {3:2.4f} Time = {4:2.4f} seconds \n'.format(im, psnr_x_,
                                                                                                     ssim_x_, deltaE_x_, elapsed_time))

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        deltaE_avg = np.mean(deltaEs)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        print()
        log('Dataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}, Avg. DeltaE = {3:2.4f}'.format(set_cur, psnr_avg, ssim_avg, deltaE_avg))

        if not args.save_visual_only:
            fsrgb.write('\nDataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}, Avg. DeltaE = {3:2.4f}'.format(set_cur, psnr_avg, ssim_avg, deltaE_avg))
            fsrgb.close()
