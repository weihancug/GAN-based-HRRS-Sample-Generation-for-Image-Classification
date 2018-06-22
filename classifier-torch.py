import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torchvision
import math
import time
import torch.backends.cudnn as cudnn
BATCH_SIZE = 128
LR = 0.01
nlabels = 21
EPOCH = 300
DATA_DIR = '/home/hpc-126/remote-host/UCM/train/'
trainLoader = ''
testLoader = ''
Training = True
momentum = 0.9
gamma =0.1
weight_decay = 5e-4
print_freq = 10
model = ''
#model = '/home/hpc-126/remote-host/WGAN-Tensorflow/classifier_model/vgg_epoch_99.pth'
gpu=0
traindir = os.path.join(DATA_DIR,'train/')
#traindir = ''
testdir = os.path.join(DATA_DIR,'test/')
generated_traindir = '/home/hpc-126/remote-host/UCM/generatetrain'
#generated_traindir = ''
print traindir
print testdir



#vgg 9
cfg = [64,'M',128,'M',256,'M',512,512,'M',512,512,'M']
   # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
 #   [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class _VGG (nn.Module):
    def __init__(self,cfg):
        super(_VGG, self).__init__()
        self.features = make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4*4, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, nlabels),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward (self,input):
        output = self.features(input)
        output = output.view(input.size(0),-1)
        output = self.classifier(output)
        return output

def main():
    cudnn.benchmark = True
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    GtrainLoader = ''
    if generated_traindir !='':
        GtrainLoader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(generated_traindir,data_transforms),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
    trainLoader = ''
    if traindir !='':
        trainLoader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(traindir, data_transforms),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

    testLoader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, data_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    vgg = _VGG(cfg)
    print (vgg)
    # load  pre-train model
    if model !='' :
        vgg.load_state_dict(torch.load(model))
    vgg.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(vgg.parameters(), lr=LR
                                    )
#    , momentum = momentum, weight_decay = weight_decay
    #training
    for epoch in range (EPOCH):
        #training the model

        adjust_learning_rate(optimizer,epoch)
        if trainLoader =='' and GtrainLoader =='':
            print ('please specify the train set ')
        if trainLoader !='':
            print ('train model with train data')
            train(trainLoader,vgg,criterion,optimizer,epoch)
        if GtrainLoader !='' and generated_traindir !='':
            print ('train model with generated data')
            train(GtrainLoader, vgg, criterion, optimizer, epoch)
        print ('test model with validation set')
        prec1 = validate(testLoader,vgg,criterion)
    #     for step, (images, labels) in enumerate(trainLoader):
    # #  for images, labels in trainLoader:
    #         step+=1
    #         images = Variable(images).cuda(gpu)
    #         labels = Variable(labels).cuda(gpu)
    #
    #         outputs = vgg(images)
    #        # print(outputs)
    #         loss = criterion(outputs,labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print ('EPOCH [%4d/%4d], STEP[%4d/%4d] loss on training set: %.4f, ' % (epoch, EPOCH,step,len(trainLoader),loss.data[0]))

        # testing classification accuracy on test set
        # correct = 0
        # total = 0
        # for images,labels in testLoader:
        #     images =Variable(images).cuda(gpu)
        #     outputs = vgg(images)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted.cpu()==labels).sum()
        # print ('total:%d'%total)
        # print ('correct:%d'%correct)
        # print ('test accuracy on test accuracy is %f%%'%(100*float(correct)/float(total)))
        if epoch!=0 and (epoch+1)%100==0:
            torch.save(vgg.state_dict(),'/home/hpc-126/remote-host/WGAN-Tensorflow/classifier_model/vgg_epoch_%d.pth'%(epoch))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(gpu,async=True)
        input_var = torch.autograd.Variable(input).cuda(gpu)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # print target_var
        # print output
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  ' {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DatTimea {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(gpu,async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda(gpu)
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.3 ** (epoch //100))
    print ('learning rate is :%f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == '__main__':
    main()
