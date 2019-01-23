from __future__ import print_function
import argparse
from math import log10
import os
from os import errno

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from model import SRCNN

parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)


use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if use_cuda:
#     torch.cuda.manual_seed(opt.seed)


train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


srcnn = SRCNN()
criterion = nn.MSELoss()


if(use_cuda):
	srcnn.cuda()
	criterion = criterion.cuda()

optimizer = optim.SGD(srcnn.parameters(),lr=opt.lr)
#optimizer = optim.Adam(srcnn.parameters(),lr=opt.lr)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        model_out = srcnn(input)
        loss = criterion(model_out, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))




def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = srcnn(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(srcnn, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.epochs + 1):
    train(epoch)
    test()
    if(epoch%10==0):
        checkpoint(epoch)
