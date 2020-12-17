import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import visdom
import wandb
from torch.utils.data import DataLoader

from dataset import *
from model import IDCN
from utils import *

parser = argparse.ArgumentParser(description="PyTorch IDCN")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--dataset", default='datasets/train_jpg/', type=str, help="dataset path")
parser.add_argument("--checkpoints_path", default='checkpoints/', type=str, help="checkpoints path")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=20000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1600,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=200")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--qf", default=10, type=int, help="qf")
parser.add_argument("--device", default=0, type=int, help="which gpu to use")
parser.add_argument("--visualization", default='wandb', type=str, help="none or wandb or visdom")
opt = parser.parse_args()

min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0

if opt.visualization == 'visdom':
    vis = visdom.Visdom(env='IDCN')


def main():
    global opt, model

    torch.cuda.set_device(opt.device)

    if opt.visualization == 'wandb':
        wandb.init(project="IDCN")

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = TrainDataset(dir=opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                      shuffle=True, num_workers=opt.threads)

    print("===> Building model")
    pyramid_cells = (3, 2, 1, 1, 1, 1)
    qy = get_table(luminance_quant_table, opt.qf)
    qc = get_table(chrominance_quant_table, opt.qf)
    model = IDCN(n_channels=64, n_pyramids=8,
                 n_pyramid_cells=pyramid_cells, n_pyramid_channels=64, qy=qy, qc=qc)

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if opt.visualization == 'wandb':
        wandb.watch(model)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 1
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    # lr = opt.lr
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    print('lr{}  iter:'.format(lr, n_iter))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter

    avr_loss = 0

    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = batch[0], batch[1]  # b c h w

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                            loss.item()))
    avr_loss = avr_loss / len(training_data_loader)

    if opt.visualization == 'wandb':
        wandb.log({"Test avr_loss": avr_loss})
    elif opt.visualization == 'visdom':
        vis.line(Y=np.array([avr_loss]), X=np.array([epoch]),
                 win='loss',
                 opts=dict(title='loss'),
                 update='append'
                 )

    epoch_avr_loss = avr_loss
    if epoch_avr_loss < min_avr_loss:
        min_avr_loss = epoch_avr_loss
        print('|||||||||||||||||||||min_epoch_loss is {:.10f}|||||||||||||||||||||'.format(min_avr_loss))
        save_flag = True
    else:
        save_flag = False
        print('epoch_avr_loss is {:.10f}'.format(epoch_avr_loss))


def save_checkpoint(model, epoch):
    global min_avr_loss
    global save_flag

    model_folder = opt.checkpoints_path
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    if (epoch % 10) == 0:
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    # if save_flag is True:
    #     torch.save(model, '{}epoch_{}_min_batch_loss_{}.pth'.format(model_folder, epoch, min_avr_loss))
    #     print('min_loss model saved')


if __name__ == "__main__":
    main()
