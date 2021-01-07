from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import SurrogateModel
from logger import Logger
import torch.nn as nn
import os
import argparse
import pdb

def parser():
    parser = argparse.ArgumentParser(description='SurrogateModel Training')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Loading pretrain data')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epoch', '-e', default=None, help='resume from epoch')
    parser.add_argument('--Epochs', '-E', default=50, help='the num of training epochs')
    parser.add_argument('--LR', default=0.01, help='Learning Rate')
    args = parser.parse_args()
    return args

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x, Max, Min

def DataLoad ():
    D = np.loadtxt('./cache/7.txt')
    x = D[:, 0:-1]
    y = D[:, -1].reshape(10000,-1)
    pdb.set_trace()
    # x = np.loadtxt('./cache/7x.txt')
    # y = np.loadtxt('./cache/7y.txt').reshape(10000, -1)
    # x, _, _ = MaxMinNormalization(x)
    y, ymax, ymin = MaxMinNormalization(y)
    x = x / 255
    # pdb.set_trace()
    X = torch.from_numpy(x).float()
    Y = torch.from_numpy(y).float()
    Dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset=Dataset, batch_size=100, shuffle=True, num_workers=2)
    return train_loader, X, Y, ymax, ymin

def train (epoch, train_loader, net, Loss, args, log, ymax, ymin):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = args.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 阶梯式衰减
    net.train()
    sum_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        length = len(train_loader)
        # pdb.set_trace()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        out = net(inputs)
        out1 = out * (ymax - ymin) + ymin
        # pdb.set_trace()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        # pdb.set_trace()
    log.logger.info('第%d个epoch的损失为%f\n' % (epoch + 1, sum_loss/length))
    Loss.append(sum_loss / length)
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        state = {
            'net': net.state_dict(),
            'epoch': epoch
        }
        if not os.path.isdir('./cache/checkpoint/'+'pic7'):
            os.mkdir('./cache/checkpoint/'+'pic7')
        torch.save(state, './cache/checkpoint/'+'pic7'+'/'+str(epoch+1)+ 'ckpt.pth')

def visualization(Loss, y, y_pre):
    plt.figure(1)
    plt.title("loss")
    plt.plot(Loss)
    plt.show()

    plt.figure(2)
    plt.title("curve")
    y, = plt.plot(y)
    y_pre, = plt.plot(y_pre)
    plt.legend([y, y_pre], ["y", "y_pre"])
    plt.show()
    return None

def main():
    args = parser()
    log = Logger('./cache/log/' + '7_trainlog.txt', level='info')
    train_loader, X, Y, ymax, ymin = DataLoad()
    start_epoch = 0
    Loss = []
    net = SurrogateModel().cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    torch.backends.cudnn.benchmark = True
    if args.resume:
        log.logger.info('Resuming from checkpoint')
        weighted_file = os.path.join('./cache/checkpoint/' + 'pic7', args.epoch + 'ckpt.pth')
        checkpoint = torch.load(weighted_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
    for epoch in range(start_epoch, args.Epochs):
        train(epoch, train_loader, net, Loss, args, log ,ymax, ymin)
    log.logger.info('='*80)
    log.logger.info(">>Training Finished!<<  Total EPOCH=%d\n" % args.Epochs)
    y_pre = net(X).cuda().data.cpu().numpy()
    y = Y.numpy()
    log.logger.info(y_pre)
    # visualization(Loss, y, y_pre)

if __name__ == '__main__':
    main()
