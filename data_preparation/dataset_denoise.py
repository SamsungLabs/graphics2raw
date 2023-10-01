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

# no need to run this code separately

import os
import torch
from torchvision import transforms
import data_preparation.data_generator_denoise as dg
from noise_profiler.image_synthesizer import load_noise_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def numpy2tensor(ims, preload_on_cuda):
    ims = torch.from_numpy(ims)
    ims = ims.permute(0, 3, 1, 2)
    if preload_on_cuda:
        ims = ims.to(device)
    return ims


class DatasetRAW(object):
    def __init__(self, root, batch_size, patch_size, stride, preload_on_cuda, iso=3200, data_type='graphics_raw'):
        print('Inside raw data generator')

        noise_model_path = './noise_profiler/h-gauss-s20-v1'
        noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_noise_model(path=noise_model_path)
        noise_model_obj = {
            'noise_model': noise_model,
            'iso2b1_interp_splines': iso2b1_interp_splines,
            'iso2b2_interp_splines': iso2b2_interp_splines,
        }

        input_ims, target_ims = dg.datagenerator_raw(data_dir=os.path.join(root),
                                                     batch_size=batch_size,
                                                     patch_size=patch_size, stride=stride,
                                                     data_type=data_type,
                                                     noise_model=noise_model_obj, iso=iso)
        print('Number of patches ' + str(input_ims.shape[0]))
        print('Outside raw data generator \n')

        self.input_ims = numpy2tensor(input_ims, preload_on_cuda)
        self.target_ims = numpy2tensor(target_ims, preload_on_cuda)

        assert patch_size % 2 == 0  # crop stride must be even to preserve the bayer pattern

    def __getitem__(self, idx):
        # load images
        img = self.input_ims[idx]  # c, h, w
        target = self.target_ims[idx]
        return img, target

    def __len__(self):
        return len(self.input_ims)


if __name__ == '__main__':
    from utils.torch_utils import torchvision_visualize_raw
    data_dir = 'graphics_dataset'
    batch_size, patch_size, stride = 16, 64, 64
    preload_on_cuda = False

    image_datasets = {
        x: DatasetRAW(os.path.join(data_dir, x), batch_size, patch_size, stride, preload_on_cuda)
        for x in ['val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

    # Get a batch of training data
    inputs, targets = next(iter(dataloaders['val']))  # b, c, h, w;

    img_grid = torchvision_visualize_raw([inputs, targets])
    img_grid = transforms.ToPILImage()(img_grid.cpu())
    img_grid.save("debug_inputs_targets_grid.png")
    print('Done!')
