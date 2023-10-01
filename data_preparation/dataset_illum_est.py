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

import torch
import torchvision
from torchvision import transforms
import data_preparation.data_generator_illum_est as dg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetIllumEst(object):
    def __init__(self, dataset_dir, illum_file, split_file, split, batch_size, patch_size, stride, on_cuda):

        print('Inside illum est data generator')
        in_patches, gt_illums = dg.datagenerator_illum_est(dataset_dir=dataset_dir,
                                                           illum_file=illum_file,
                                                           split_file=split_file,
                                                           split=split,
                                                           batch_size=batch_size,
                                                           patch_size=patch_size,
                                                           stride=stride)

        in_patches = torch.from_numpy(in_patches)
        in_patches = in_patches.permute(0, 3, 1, 2)  # b, c, h, w
        gt_illums = torch.from_numpy(gt_illums)  # b, 3
        if on_cuda:
            in_patches = in_patches.to(device)
            gt_illums = gt_illums.to(device)

        print('Number of patches ' + str(in_patches.shape[0]))
        print('Number of illums ' + str(gt_illums.shape[0]))
        print('Outside illum est data generator \n')
        self.in_patches = in_patches
        self.gt_illums = gt_illums

    def __getitem__(self, idx):
        # load images
        img = self.in_patches[idx]
        gt_illum = self.gt_illums[idx]

        return img, gt_illum

    def __len__(self):
        return len(self.in_patches)


if __name__ == '__main__':
    image_datasets = {
        x: DatasetIllumEst(dataset_dir='illum_est_expts/data/ours',
                           illum_file='illum_est_expts/data/ours/gt_illum.p',
                           split_file='illum_est_expts/data/SamsungNX2000_train_valid_test_split_idx.p',
                           split=x,
                           batch_size=32,
                           patch_size=48,
                           stride=48,
                           on_cuda=True)
        for x in ['val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=0)
                   for x in ['val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

    # Get a batch of training data
    inputs, targets = next(iter(dataloaders['val']))

    inputs = inputs ** (1 / 2.2)
    target_ims = targets[..., None, None] * torch.ones((32, 3, 48, 48))  # 32, 3, 1, 1 * 32, 3, 48, 48
    # Make a grid from batch
    targets_grid = torchvision.utils.make_grid(target_ims)
    inputs_grid = torchvision.utils.make_grid(inputs)

    targets_grid = transforms.ToPILImage()(targets_grid.cpu())
    inputs_grid = transforms.ToPILImage()(inputs_grid.cpu())

    targets_grid.save("debug_targets_grid.png")
    inputs_grid.save("debug_inputs_grid.png")

    print('Done!')
