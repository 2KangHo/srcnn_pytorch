from __future__ import print_function
import argparse
from math import log10
import time
import os
from os import errno
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_validation_set, get_test_set
from model import SRCNN

parser = argparse.ArgumentParser(
    description='PyTorch SRCNN')
parser.add_argument('--upscale_factor', type=int, default=2,
                    required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64,
                    help='training batch size')
parser.add_argument('--test_batch_size', type=int,
                    default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16,
                    help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
parser.add_argument('--add_noise', action='store_true',
                    help='add gaussian noise?')
parser.add_argument('--noise_std', type=float, default=3.0,
                    help='standard deviation of gaussian noise')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to test or resume model')


def main():
    global opt
    opt = parser.parse_args()
    opt.gpuids = list(map(int, opt.gpuids))

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    cudnn.benchmark = True

    train_set = get_training_set(
        opt.upscale_factor, opt.add_noise, opt.noise_std)
    validation_set = get_validation_set(opt.upscale_factor)
    test_set = get_test_set(opt.upscale_factor)
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    validating_data_loader = DataLoader(
        dataset=validation_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

    model = SRCNN()
    criterion = nn.MSELoss()

    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.test:
        model_name = join("model", opt.model)
        model = torch.load(model_name)
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])
        start_time = time.time()
        test(model, criterion, testing_data_loader)
        elapsed_time = time.time() - start_time
        print("===> average {:.2f} image/sec for processing".format(
            100.0/elapsed_time))
        return

    for epoch in range(1, opt.epochs + 1):
        train(model, criterion, epoch, optimizer, training_data_loader)
        validate(model, criterion, validating_data_loader)
        if epoch % 10 == 0:
            checkpoint(model, epoch)


def train(model, criterion, epoch, optimizer, training_data_loader):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        model_out = model(input)
        loss = criterion(model_out, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def validate(model, criterion, validating_data_loader):
    avg_psnr = 0
    for batch in validating_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1.0 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(validating_data_loader)))


def test(model, criterion, testing_data_loader):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1.0 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))


def checkpoint(model, epoch):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    main()
