import torch.optim as optim
from PIL import Image
import torch
import os
import random
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import Counter
import datetime
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import scipy.signal
from scipy.interpolate import make_interp_spline, BSpline
import scipy.interpolate as interpolate
import pandas as pd
from torch.utils import data

    


def index_split(full_or_no, csv_path):
    csv_path = 'chipinfo.csv' 
    df = pd.read_csv(csv_path)
    training = df.loc[df['depression'] == 17]
    subclass_9 = training.loc[training['target_type'] != 'bmp2_tank']
    subclass_8 = subclass_9.loc[subclass_9['target_type'] != 't72_tank'].index.values
    class_1_train = np.array(training.loc[training['serial_num']=='c21'].index.values)
    class_3_train = np.array(training.loc[training['serial_num']=='132'].index.values)
    subclass = np.concatenate([subclass_8, class_1_train, class_3_train], axis=0)
    training = training.index
    testing = df.loc[df['depression'] == 15]
    subclass_test9 = testing.loc[testing['target_type'] != 'bmp2_tank']
    subclass_test8 = np.array(subclass_test9.loc[subclass_test9['target_type']=='t72_tank'].index.values)
    class_1_test2 = np.array(testing.loc[testing['serial_num']=='9563'].index.values)
    class_1_test3 = np.array(testing.loc[testing['serial_num']=='9566'].index.values)
    class_3_test2 = np.array(testing.loc[testing['serial_num']=='812'].index.values)
    class_3_test3 = np.array(testing.loc[testing['serial_num']=='s7'].index.values)
    subclass_test = np.concatenate([subclass_test8, class_1_test2, class_1_test3, class_3_test2, class_3_test3], axis=0)
    testing = np.array(testing.index.values)
        
    if full_or_no:
        return training, testing
    else:
        return subclass, subclass_test
    
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def data_prep_11(batch_size, split_method, data_dir, csv_path):
    #split_method: True: 15/17 angle depression
    #              False: seen/unseen split
    data_x = []
    data_y = []
    for f in listdir(data_dir):
        data = np.load(join(data_dir, f))
        label = f.split('_')[0].split('c')[1]
        data_x.append(data)
        data_y.append(int(label)-1)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    xshape = data_x.shape
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))

    #Indicate which splitting method
    train_idx, test_idx = index_split(split_method, csv_path)
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    
    params_train = {
          'shuffle': True,
          'num_workers': 1}

    params_val = {'batch_size': 400,
              'shuffle': False,
              'num_workers': 1}
    
    train_generator = torch.utils.data.DataLoader(dataset=data_train, batch_size = batch_size, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    
    return train_generator, test_generator 



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
    
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / len(train_loader)
            
            
def eval_train(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    real_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
             
    test_loss /= len(test_loader.dataset)
    print('\nTraining set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))   
    return 100. * correct / len(test_loader.dataset)
            
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(test_loader.dataset)



def draw_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    matrix = cm > 0
    final = np.zeros((cm.shape[0], cm.shape[1], 3))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == j:
                final[i][j] = [84 / 255., 118 / 255., 33 / 255.]
            elif matrix[i][j] != 0:
                final[i][j] = [208 / 255., 123 / 255., 12 / 255.]
            else:
                final[i][j] = [1., 1., 1.]
            
    fig, ax = plt.subplots()
    im = ax.imshow(final, interpolation='nearest')
#     ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='MSTAR Dataset Classification',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                if cm[i, j] < 0.01:
                    cm[i, j] = 0.01
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('temp.png', dpi=fig.dpi)
    fig.savefig('temp.eps', dpi=fig.dpi, format='eps')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet50 Example on MSTAR')
    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batchsize', type=int, default=30, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Adam momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42222222, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', type=str, default='./data_r2', metavar='N',
                        help='training and testing data directory')
    parser.add_argument('--csv-dir', type=str, default='./chipinfo.csv', metavar='N',
                        help='chipinfo directory for splitting')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda: 0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=11).to(device) 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-3, amsgrad=True )#momentum=args.momentum)

    print("#### PROCESSING DATA ####")
    train_loader, test_loader = data_prep_11(args.batchsize,False,args.data_dir,args.csv_dir)
    print("#### END DATAPROCESSING ####")
 
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        acc = eval_train(args, model, device, train_loader)
        testacc = test(args, model, device, test_loader)
        print('[epoch {}] TrainAccuracy: [{}] TrainLoss: [{}]'.format(epoch, acc, loss))
        print('[epoch {}] TestAccuracy: [{}]'.format(epoch, testacc))

if __name__ == '__main__':
    main()
