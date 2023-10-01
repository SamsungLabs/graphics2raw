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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data_preparation.dataset_illum_est import DatasetIllumEst
from torch.optim import lr_scheduler
import time
import os
import argparse
from torch.utils.tensorboard import SummaryWriter


def parse_args():

    parser = argparse.ArgumentParser(description='Day-to-night train')
    parser.add_argument(
        '--dataset-dir', default='illum_est_expts/data/ours', type=str, help='folder of png images')
    parser.add_argument(
        '--illum_file', default='illum_est_expts/data/ours/gt_illum.p', type=str,
        help='path to split indices')
    parser.add_argument(
        '--split_file', default='illum_est_expts/data/SamsungNX2000_train_valid_test_split_idx.p', type=str,
        help='path to split indices')
    parser.add_argument(
        '--savefoldername', default='models', type=str, help='folder to save trained models to')
    parser.add_argument(
        '--exp_dir', default='./', type=str, help='directory to save experiment data to')
    parser.add_argument(
        '--patch-size', type=int, default=64, help='patch size')
    parser.add_argument(
        '--stride', type=int, default=64, help='stride when cropping patches')
    parser.add_argument(
        '--batch-size', type=int, default=128, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--milestones', default='', type=str, help='milestones as comma separated string')
    parser.add_argument(
        '--num-epochs', type=int, default=500, help='number of epochs')
    parser.add_argument(
        '--num_filters', type=int, default=7, help='number of filters for CNN layers ')
    parser.add_argument(
        '--tboard-freq', type=int, default=200, help='frequency of writing to tensorboard')
    parser.add_argument('--on-cuda', default=False, action='store_true', help='False: load each batch on cuda, True: load all data directly on cuda')
    parser.add_argument(
        '--model_save_freq', type=int, default=50, help='save model per model_save_freq epochs')
    parser.add_argument(
        '--tb_im_save_freq', type=int, default=500, help='save tensorboard images per tb_im_save_freq epochs')
    args = parser.parse_args()
    print(args)

    return args


EPS = 1e-9
PI = 22.0 / 7.0


def angular_loss(predicted, gt, shrink=True):
    predicted = predicted.reshape((-1, 3))
    gt = gt.reshape((-1, 3))
    cossim = torch.clamp(torch.sum(predicted * gt, dim=1) / (
            torch.norm(predicted, dim=1) * torch.norm(gt, dim=1) + EPS), -1, 1.)
    if shrink:
        angle = torch.acos(cossim * 0.9999999)
    else:
        angle = torch.acos(cossim)
    a_error = 180.0 / PI * angle
    a_error = torch.sum(a_error) / a_error.shape[0]
    return a_error


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):

    milestones = [int(item) for item in args.milestones.split(',')] if args.milestones else []

    savefoldername = args.savefoldername
    exp_dir = args.exp_dir
    tb_dir = os.path.join(exp_dir, savefoldername, 'tensorboard')

    os.makedirs(tb_dir, exist_ok=True)
    writers = [SummaryWriter(os.path.join(tb_dir, savefoldername))]

    modsavepath = os.path.join(exp_dir, savefoldername, 'models')
    if not (os.path.exists(modsavepath) and os.path.isdir(modsavepath)):
        os.makedirs(modsavepath)

    save_args(args, modsavepath)

    image_datasets = {
            x: DatasetIllumEst(dataset_dir=args.dataset_dir,
                               illum_file=args.illum_file,
                               split_file=args.split_file,
                               split=x,
                               batch_size=args.batch_size,
                               patch_size=args.patch_size,
                               stride=args.stride,
                               on_cuda=args.on_cuda)
            for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    epoch_loss = {x: 0.0 for x in ['train', 'val']}
    epoch_ang_err = {x: 0.0 for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = IllumEstNet(in_channels=3, out_channels=3, num_filters=args.num_filters)
    num_params = count_parameters(model)
    model = model.to(device)

    criterion = nn.L1Loss()

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # training loop starts here
    since = time.time()

    best_loss = 10 ** 6
    best_ang_err = 0.0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        running_loss_tboard = 0.0
        running_ang_err_tboard = 0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_ang_err = 0.0

            # Iterate over data.
            for i, (inputs, targets) in enumerate(dataloaders[phase]):

                if not args.on_cuda:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    ang_err_out = angular_loss(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        running_loss_tboard += loss.item()
                        running_ang_err_tboard += ang_err_out.item()
                        if (i+1) % args.tboard_freq == 0:  # every tboard_freq mini-batches...

                            # ...log the running loss
                            for writer in writers:
                                writer.add_scalar('loss',
                                                  running_loss_tboard / args.tboard_freq,
                                                  epoch * len(dataloaders[phase]) + i)

                                writer.add_scalar('ang_err',
                                                  running_ang_err_tboard / args.tboard_freq,
                                                  epoch * len(dataloaders[phase]) + i)

                            running_loss_tboard = 0.0
                            running_ang_err_tboard = 0.0

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_ang_err += ang_err_out.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_ang_err[phase] = running_ang_err / dataset_sizes[phase]

            if phase == 'val':
                outputs_im = outputs[..., None, None] * torch.ones((args.batch_size, 3, args.patch_size,
                                                                    args.patch_size)).to(device)
                target_ims = targets[..., None, None] * torch.ones((args.batch_size, 3, args.patch_size,
                                                                    args.patch_size)).to(device)

                img_grid = torchvision.utils.make_grid(torch.cat((inputs[:, 0:3, :, :], outputs_im,
                                                                  target_ims), 2), normalize=True, range=(0, 1))
                # ...log the running loss
                for writer in writers:
                    writer.add_scalars('loss',
                                       {'train': epoch_loss['train'], 'val': epoch_loss['val']},
                                       (epoch + 1) * len(dataloaders['train']))

                    writer.add_scalars('ang_err',
                                       {'train': epoch_ang_err['train'], 'val': epoch_ang_err['val']},
                                       (epoch + 1) * len(dataloaders['train']))
                    if (epoch + 1) % args.tb_im_save_freq == 0:
                        writer.add_image('val_epoch_' + str(epoch), img_grid)

            print('{} Loss: {:.6f} Ang Err: {:.4f}'.format(
                phase, epoch_loss[phase], epoch_ang_err[phase]))

            # save the model
            if phase == 'val' and epoch_loss[phase] < best_loss:
                best_loss = epoch_loss[phase]
                best_ang_err = epoch_ang_err[phase]
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(modsavepath, 'bestmodel.pt'))

        # save the model
        if (epoch + 1) % args.model_save_freq == 0:
            torch.save(model.state_dict(), os.path.join(modsavepath, f'model_ep_{epoch + 1}.pt'))

        print()

    time_elapsed = time.time() - since
    train_str = 'Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60)
    train_str += 'Best val loss: {:4f}\n'.format(best_loss)
    train_str += 'Best val ang err: {:4f}\n'.format(best_ang_err)
    train_str += 'Best epoch: {}\n'.format(best_epoch)
    train_str += f'Number of model trainable parameters: {num_params}\n'
    with open(os.path.join(modsavepath, 'train_info.txt'), 'w') as f:
        f.write(train_str)


if __name__ == "__main__":

    args = parse_args()
    main(args)
