import os
import argparse
import logging
import time
import json
import librosa
import matplotlib.pyplot as plt
import librosa.display as display
from tqdm import tqdm
from PIL import Image
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=11)  # 11/12/21/22
parser.add_argument('--wav', type=str, default='./dataset_new/task1_wav/' )
parser.add_argument('--input', type=str, default='./dataset_new/input/')
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--size_s', type=int, default=128)
parser.add_argument('--size_m', type=int, default=56)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--save', type=str, default='./results')
parser.add_argument('--classweight', type=str, default='true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


def wavelet(sig):
    cA, out = pywt.dwt(sig, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    A = cA
    
    for i in range(6):
        cA, cD = pywt.dwt(A, 'db8')
        A = cA
        out = np.hstack((out,cD))

    out = np.hstack((out,A))
        
    return out


def reshape(matrix):
    num = matrix.shape[0]
    length = math.ceil(np.sqrt(num))
    zero = np.zeros([np.square(length)-num,])
    matrix = np.concatenate((matrix,zero))
    out = matrix.reshape((length,length))

    return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
#        print("input:"+str(x.size()))
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.droupout(out)
#        print("output:"+str(out.size()))
        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class BilinearCNN(nn.Module):
    
    def __init__(self,dim):
        super(BilinearCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64)
        self.ResNet_6 = ResBlock(64, 64)
        self.ResNet_7 = ResBlock(64, 64)
        self.ResNet_8 = ResBlock(64, 64)
        self.ResNet_9 = ResBlock(64, 64)
        self.ResNet_10 = ResBlock(64, 64)
        self.ResNet_11 = ResBlock(64, 64)
        self.ResNet_12 = ResBlock(64, 64)
        self.ResNet_13 = ResBlock(64, 64)
        self.ResNet_14 = ResBlock(64, 64)
        self.ResNet_15 = ResBlock(64, 64)
        self.ResNet_16 = ResBlock(64, 64)
        self.ResNet_17 = ResBlock(64, 64)
        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        if args.task == 11:
            op = 2
        elif args.task == 12:
            op = 7
        elif args.task == 21:
            op = 3
        elif args.task == 22:
            op = 5

        self.linear = nn.Linear(64, op)
        self.dropout = nn.Dropout(args.dropout)
        self.flat = Flatten()
        
        
    def forward(self, stft, mfcc):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)
        out_s = self.ResNet_8(out_s)
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)
        out_s = self.ResNet_14(out_s)
        out_s = self.ResNet_16(out_s)
        out_s = self.norm0(out_s)
        out_s = self.relu0(out_s)
        out_s = self.pool0(out_s)
        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)
        out_m = self.ResNet_15(out_m)
        out_m = self.ResNet_17(out_m)
        out_m = self.norm1(out_m)
        out_m = self.relu1(out_m)
        out_m = self.pool1(out_m)

        out = torch.matmul(out_s, out_m)
        out = self.flat(out)
        out = self.linear(out)
        out = self.dropout(out)
       
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class myDataset(data.Dataset):
    def __init__(self, stft, mfcc, targets):
        self.stft = stft
        self.mfcc = mfcc
        self.targets = targets

    def __getitem__(self, index):
        sample_stft = self.stft[index]
        sample_mfcc = self.mfcc[index]
        target = self.targets[index]
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
        min_m = np.min(sample_mfcc)
        max_m = np.max(sample_mfcc)
        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor(np.array(sample_stft))
        crop_s = transforms.Resize([args.size_s, args.size_s])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
        output_mfcc = torch.FloatTensor(np.array(sample_mfcc))
        crop_m = transforms.Resize([args.size_m, args.size_m])
        img_m = transforms.ToPILImage()(output_mfcc)
        croped_img_m=crop_m(img_m)
        output_mfcc = transforms.ToTensor()(croped_img_m)

        return output_stft, output_mfcc, target

    def __len__(self):
        return len(self.targets)


def get_mnist_loaders(stft_train, mfcc_train, labels_train, stft_val, mfcc_val, labels_val, train_batch_size=128, val_batch_size=32):
    train_loader = DataLoader(
        myDataset(stft_train, mfcc_train, labels_train), batch_size=train_batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        myDataset(stft_val, mfcc_val, labels_val), batch_size=val_batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def accuracy(model, dataset_loader):
    total_correct = 0
    for stft, mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        target_class = y.numpy()
        predicted_class = np.argmax(model(stft, mfcc).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def Loss(model, dataset_loader, batch_size):
    total_loss = 0
    for stft, mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = y.to(device)
        logits = model(stft, mfcc)
        entroy = nn.CrossEntropyLoss().to(device)
        loss = entroy(logits, y).cpu().numpy()
        total_loss += loss
    return total_loss  / (len(dataset_loader.dataset)/batch_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def label_feature(stft_dir, wavelet_dir, in_json_dir):
    with open(in_json_dir,'r') as json_file:
        json_dict = json_file.read()
        anno_dict = json.loads(json_dict)
        anno_key_list = list(anno_dict.keys())

    if args.task == 11:
        for anno_key in tqdm(anno_key_list):
            label_anno = anno_dict[anno_key]
            if label_anno == 'Normal':
                anno_dict[anno_key] = 0
            elif label_anno == 'Adventitious':
                anno_dict[anno_key] = 1
            stft_wavelet(anno_key, anno_dict[anno_key], stft_dir, wavelet_dir)

    elif args.task == 12:
        for anno_key in tqdm(anno_key_list):
            label_anno = anno_dict[anno_key]
            if label_anno == 'Normal':
                anno_dict[anno_key] = 0
            elif label_anno == 'Rhonchi':
                anno_dict[anno_key] = 1
            elif label_anno == 'Wheeze':
                anno_dict[anno_key] = 2
            elif label_anno == 'Stridor':
                anno_dict[anno_key] = 3
            elif label_anno == 'Coarse Crackle':
                anno_dict[anno_key] = 4
            elif label_anno == 'Fine Crackle':
                anno_dict[anno_key] = 5
            elif label_anno == 'Wheeze+Crackle':
                anno_dict[anno_key] = 6
            stft_wavelet(anno_key, anno_dict[anno_key], stft_dir, wavelet_dir)

    elif args.task == 21:
        for anno_key in tqdm(anno_key_list):
            label_anno = anno_dict[anno_key]
            if label_anno == 'Normal':
                anno_dict[anno_key] = 0
            elif label_anno == 'Adventitious':
                anno_dict[anno_key] = 1
            elif label_anno == 'Poor Quality':
                anno_dict[anno_key] = 2
            stft_wavelet(anno_key, anno_dict[anno_key], stft_dir, wavelet_dir)

    elif args.task == 22:
        for anno_key in tqdm(anno_key_list):
            label_anno = anno_dict[anno_key]
            if label_anno == 'Normal':
                anno_dict[anno_key] = 0
            elif label_anno == 'CAS':
                anno_dict[anno_key] = 1
            elif label_anno == 'DAS':
                anno_dict[anno_key] = 2
            elif label_anno == 'CAS & DAS':
                anno_dict[anno_key] = 3
            elif label_anno == 'Poor Quality':
                anno_dict[anno_key] = 4
            stft_wavelet(anno_key, anno_dict[anno_key], stft_dir, wavelet_dir)

       
def stft_wavelet(wav, key, stft_dir, wavelet_dir):
    sig, fs = librosa.load(os.path.join(wav_dir, wav),sr=8000)

    # stft
    stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
    display.specshow(librosa.amplitude_to_db(np.abs(stft),ref=np.max),y_axis='log',x_axis='time')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig(stft_dir + '/' + wav[0:-4] + f'_{key}.png')

    # wavelet       
    wave = wavelet(sig)
    xmax = max(wave)
    xmin = min(wave)
    wave = (255-0)*(wave-xmin)/(xmax-xmin)+0       
    wave = reshape(wave)
    display.specshow(wave)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig(wavelet_dir + '/' + wav[0:-4] + f'_{key}.png')


def pack(dir_stft, dir_wavelet):       
    feature_stft_list = []
    feature_wavelet_list = []
    label_list = [] 

    for file in os.listdir(dir_stft):
        I_stft = Image.open(dir_stft + file).convert('L')
        I_wavelet = Image.open(dir_wavelet + file).convert('L')
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)
        label = int(file[-5])

        feature_stft_list.append(I_stft)
        feature_wavelet_list.append(I_wavelet)
        label_list.append(label)

    return feature_stft_list, feature_wavelet_list, label_list


def train_val(train_loader, val_loader):
    results_path = os.path.join(args.save, f'task{args.task}')
    makedirs(results_path)
    logger = get_logger(logpath=os.path.join(results_path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboardlogs'))
    
    model = BilinearCNN(64).to(device)
    
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    if args.task == 11:
        weights = [1.2984, 4.3517] # two classes
    elif args.task == 12:
        weights = [1.2984, 151.6410, 14.7850, 422.4286, 120.6939, 7.1253, 219.0370]  # seven classes
    elif args.task == 21:
        weights = [1.5077, 4.0252, 11.3226]  # three classes
    elif args.task == 22:
        weights = [1.5077, 14.1532, 7.5000, 22.5000, 11.3226]  # five classes

    class_weights = torch.FloatTensor(weights)
    if args.classweight == 'true':
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    elif args.classweight == 'false':
        criterion = nn.CrossEntropyLoss().to(device)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.train_batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[10,30,40],
        decay_rates=[1,0.1,0.01,0.001]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    
    best_epoch = 0
    best_train_acc = 0
    best_val_acc = 0
    best_train_loss = 10000
    best_val_loss = 10000
    batch_time_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):
        model.train()
        torch.cuda.empty_cache()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        stft, mfcc, y = data_gen.__next__()
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = y.to(device)
        logits = model(stft, mfcc)     
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if (itr+1) % batches_per_epoch == 0:
            model.eval()
            with torch.no_grad():
                train_acc = accuracy(model, train_loader)
                val_acc = accuracy(model, val_loader)
                train_loss = Loss(model, train_loader, args.train_batch_size)
                val_loss = Loss(model, val_loader, args.val_batch_size)
                writer.add_scalar('train/loss', train_loss, itr // batches_per_epoch)
                writer.add_scalar('val/loss', val_loss, itr // batches_per_epoch)
                writer.add_scalar('train/acc', train_acc, itr // batches_per_epoch)
                writer.add_scalar('val/acc', val_acc, itr // batches_per_epoch)
    
                if val_acc > best_val_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(results_path, 'model.pth'))
                    best_epoch = itr // batches_per_epoch
                    best_train_acc = train_acc
                    best_val_acc = val_acc
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) |  "
                    "Train Acc {:.4f} | Val Acc {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} |  "
                    "Best Epoch {:04d} |  "
                    "Best Train Acc {:.4f} | Best Val Acc {:.4f} | Best Train Loss {:.4f} | Best Val Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, train_acc, val_acc, train_loss , val_loss,
                        best_epoch, best_train_acc, best_val_acc, best_train_loss , best_val_loss 
                    )
                )


if __name__ == '__main__':
    wav_dir = args.wav
    wav_list = os.listdir(wav_dir)
    
    stft_dir_train = os.path.join('./features/stft/train', f'task{args.task}')
    stft_dir_val = os.path.join('./features/stft/val', f'task{args.task}')
    wavelet_dir_train = os.path.join('./features/wavelet/train', f'task{args.task}')
    wavelet_dir_val = os.path.join('./features/wavelet/val', f'task{args.task}')
    in_json_dir_train = os.path.join(args.input, 'train', f'task{args.task}_input.json')
    in_json_dir_val = os.path.join(args.input, 'val', f'task{args.task}_input.json')

    label_feature(stft_dir_train, wavelet_dir_train, in_json_dir_train)
    label_feature(stft_dir_val, wavelet_dir_val, in_json_dir_val)

    # pack
    stft_train, mfcc_train, labels_train = pack(stft_dir_train, wavelet_dir_train)
    stft_val, mfcc_val, labels_val = pack(stft_dir_val, wavelet_dir_val)

    train_loader, val_loader= get_mnist_loaders(
            stft_train, mfcc_train, labels_train,
            stft_val, mfcc_val, labels_val,
            args.train_batch_size, args.val_batch_size
        )
    
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_val(train_loader, val_loader)
