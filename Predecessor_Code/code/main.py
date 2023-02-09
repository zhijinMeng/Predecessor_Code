import os
import argparse
import json
import scipy.io.wavfile as wav
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


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=11)  # 11/12/21/22
parser.add_argument('--wav', type=str, default='./dataset_new/task1_wav/test/' )
parser.add_argument('--out', type=str, default='./results/task11/task11_output.json')
parser.add_argument('--test_batch_size', type=int, default=32)
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


def pack(dir_stft, dir_wavelet):       
    feature_stft_list=[]
    feature_wavelet_list=[]
    for file in os.listdir(dir_stft):
        I_stft = Image.open(dir_stft+'/'+file).convert('L')
        I_wavelet = Image.open(dir_wavelet+'/'+file).convert('L')
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)

        feature_wavelet_list.append(I_wavelet)
        feature_stft_list.append(I_stft)

    return feature_stft_list, feature_wavelet_list


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
        self.droupout = nn.Dropout(0.3)
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
        self.dropout = nn.Dropout(0.3)
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


class myDataset(data.Dataset):
    def __init__(self, stft, mfcc):
        self.stft = stft
        self.mfcc = mfcc

    def __getitem__(self, index):
        sample_stft = self.stft[index]
        sample_mfcc = self.mfcc[index]
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
        min_m = np.min(sample_mfcc)
        max_m = np.max(sample_mfcc)
        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor(np.array(sample_stft))
        crop_s = transforms.Resize([128, 128])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
        output_mfcc = torch.FloatTensor(np.array(sample_mfcc))
        crop_m = transforms.Resize([56, 56])
        img_m = transforms.ToPILImage()(output_mfcc)
        croped_img_m=crop_m(img_m)
        output_mfcc = transforms.ToTensor()(croped_img_m)

        return output_stft, output_mfcc

    def __len__(self):
        return len(self.stft)


def get_mnist_loaders(stft_test, mfcc_test, test_batch_size=32):
    test_loader = DataLoader(
        myDataset(stft_test, mfcc_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return test_loader


if __name__ == '__main__':
    wav_dir = args.wav
    wav_list = os.listdir(wav_dir)
    stft_dir = os.path.join('./features/stft/test', 'task' + str(args.task))
    if not os.path.exists(stft_dir):
        os.makedirs(stft_dir)
    wavelet_dir = os.path.join('./features/wavelet/test', 'task' + str(args.task))
    if not os.path.exists(wavelet_dir):
        os.makedirs(wavelet_dir)

    for i, wav in enumerate(tqdm(wav_list)):
        # stft
        sig, fs = librosa.load(os.path.join(wav_dir, wav),sr=8000)
        stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
        display.specshow(librosa.amplitude_to_db(np.abs(stft),ref=np.max),y_axis='log',x_axis='time')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig(stft_dir+'/'+wav[0:-4]+'.png')

        # wavelet       
        wave = wavelet(sig)
        xmax=max(wave)
        xmin=min(wave)
        wave=(255-0)*(wave-xmin)/(xmax-xmin)+0       
        wave = reshape(wave)
        display.specshow(wave)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig(wavelet_dir+'/'+wav[0:-4]+'.png')

    # pack
    stft_p, wavelet_p = pack(stft_dir, wavelet_dir)
    # joblib.dump((p_stft, p_wavelet), open(pack_dir+args.task+'_test.p', 'wb'))

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')   
    model = BilinearCNN(64).to(device)
    checkpoint = torch.load('./results/task'+str(args.task)+'/model.pth')#, map_location=device
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    test_loader = get_mnist_loaders(stft_p, wavelet_p, args.test_batch_size)
    label_list = []

    for stft, mfcc in tqdm(test_loader):
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        predicted_class = np.argmax(model(stft,mfcc).cpu().detach().numpy(), axis=1)
        label_list.extend(predicted_class.tolist())
    label_list = str(label_list)

    if args.task == 11:
        label_list = label_list.replace('0','Normal')
        label_list = label_list.replace('1','Adventitious')
    elif args.task == 12:
        label_list = label_list.replace('0','Normal')
        label_list = label_list.replace('1','Rhonchi')
        label_list = label_list.replace('2','Wheeze')
        label_list = label_list.replace('3','Stridor')
        label_list = label_list.replace('4','Coarse Crackle')
        label_list = label_list.replace('5','Fine Crackle')
        label_list = label_list.replace('6','Wheeze+Crackle')
    elif args.task == 21:
        label_list = label_list.replace('0','Normal')
        label_list = label_list.replace('1','Adventitious')
        label_list = label_list.replace('2','Poor Quality')
    elif args.task == 22:
        label_list = label_list.replace('0','Normal')
        label_list = label_list.replace('1','CAS')
        label_list = label_list.replace('2','DAS')
        label_list = label_list.replace('3','CAS & DAS')
        label_list = label_list.replace('4','Poor Quality')
    
    label_list = label_list.strip('[').strip(']').split(',')
    wav_label_dict = dict(zip(wav_list, label_list))
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    with open(args.out, "w") as f:
        f.write(json.dumps(wav_label_dict, ensure_ascii=False, separators=(', ',': ')))