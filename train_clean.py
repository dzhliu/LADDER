import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time

import models.model
from utils_trainclean import *


from models.vgg import *
from torchvision.models import googlenet as torch_googleNet

from torch.utils.data import Dataset

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--model_dir', type=str, default="./saved_model/")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default="resnet_ASSET",
                        help="cnn_mnist, vgg11, vgg16, preact_resnet, resnet, googlenet, densenet resnet_ASSET")       
    parser.add_argument('--data', type=str, default="cifar10",
                        help='mnist, gtsrb, cifar10, imagenet, celeba, SVHN, imagenet_full, stl10， flowers102， food101, LFWPeople')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--lr', type=float, default=0.01)  
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--num_classes', type=int, default=200)
    return parser.parse_args()


args = args_parser()
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)

class Sub_Dataset(Dataset):
    def __init__(self, data, class_mapping):
        self.data = data
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        label = self.class_mapping[label]
        return img, label

def test(model,test_loader, trigger=None, problem=None):

    model.eval()
    loss_avgmeter = AverageMeter()
    acc_avgmeter = AverageMeter()

    for batch_idx, (data, label) in enumerate(test_loader):

        
        if trigger is not None:
            for sample_idx in range(0, len(data)):
                pic1, pic2 = problem.overlap_vars_to_mat(trigger, data[sample_idx].numpy(), 'Spatial')
                data[sample_idx] = torch.tensor(pic1)
                label[sample_idx] = problem.target_label

        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc=(output.argmax(1) == label.view(-1,)).float().sum()

        loss=criterion(output,label.view(-1,))
        loss_avgmeter.update(loss.detach(),data.size(0))
        acc_avgmeter.update(batch_acc.detach(),data.size(0),True)

    model.train()

    return loss_avgmeter.avg,acc_avgmeter.avg

def main():

    save_name="surrogate"+"_"+args.model+"_"+args.data
    
    
    
    loss_avgmeter = AverageMeter()
    acc_avgmeter = AverageMeter()

    if args.data == "cifar10":
        args.num_classes = 10
    elif args.data == "gtsrb":
        args.num_classes = 43
    elif args.data == "celeba":
        args.num_classes = 8
    elif args.data == "imagenet":
        args.num_classes = 200
    elif args.data == "mnist":
        args.num_classes = 10
    elif args.data == "SVHN":
        args.num_classes = 10
    elif args.data == "imagenet_full":
        args.num_classes = 10
    elif args.data == "stl10":
        args.num_classes = 10
    elif args.data == 'flowers102':
        args.num_classes = 102
    elif args.data == 'food101':
        args.num_classes = 101

    if args.data == "imagenet_full":
        if args.model == "resnet":
            print('load torchvision.models.resnet18 with IMAGENET1K_V1 pretrained weights on imagenet task')
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        else:
            raise Exception("unknown model for imagenet_full dataset")
    else:
        if args.model=="vgg16":
            model = vgg16_bn(num_classes=args.num_classes)
        elif args.model == 'vgg19':
            model = vgg19_bn(num_classes=args.num_classes)
        elif args.model == "vgg11":
            model = vgg11_bn(num_classes=args.num_classes)
        elif args.model=="preact_resnet":
            from models.preact_resnet import PreActResNet18
            model = PreActResNet18(num_classes=args.num_classes)
        elif args.model == "cnn_mnist":
            from models.cnn_mnist import CNN_MNIST
            model = CNN_MNIST()
        elif args.model == "resnet":
            if args.data == "cifar10" or args.data == "SVHN" or args.data == 'gtsrb':
                from models.resnet_cifar10 import ResNet18
                model = ResNet18(num_classes=args.num_classes)
            elif args.data == "stl10": 
                print('load resnet18 with pretrained weights for stl10')
                model = torchvision.models.resnet18(weights = "IMAGENET1K_V1")
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                model = model.to('cuda:0')
            elif args.data == 'food101':
                print('load resnet18 from torchvision for food101')
                model = torchvision.models.resnet18()
                model.fc = torch.nn.Linear(model.fc.in_features, 101)
                model = model.to('cuda:0')
            else:
                from models.resnet import ResNet18
                model = ResNet18(num_classes=args.num_classes)
        elif args.model == "googlenet":
            from models.googlenet import GoogLeNet
            model = GoogLeNet()
        elif args.model == "densenet":
            model = torchvision.models.densenet121(weights = "IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(model.classifier.in_features, 10)
            model = model.to('cuda:0')
            print('load densenet for STL10 dataset')
        elif args.model == "resnet_ASSET":
            from models.resnet_ASSET import ResNet18 as ResNet18_ASSET
            model = ResNet18_ASSET(10)
        else:
            raise Exception('unknown model name')

    model = model.to(args.device)

    if args.data == "imagenet_full":
        
        
        trainset = torch.load('./data/train_sub_imagenet_resize72crop64_10class.pt')
        testset = torch.load('./data/test_sub_imagenet_resize72crop64_10class.pt')
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        print("a pre-selected imagenet full dataset (10 classes) has been loaded!")
    else:
        train_dataset, test_dataset = load_dataset(args, preprocess=True)
        train_loader,test_loader = load_data(args, train_dataset, test_dataset)

    

    counter = 0
    best_acc=0
    test_loss_min = 0
    lr = args.lr
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()

        if counter / 10 == 1:
            counter = 0
            lr = lr * 0.5

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        for batch_idx, (data, label) in enumerate(train_loader):

            data = data.to(args.device)
            label = label.to(args.device)

            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output,label.view(-1,))
            loss.backward()
            optimizer.step()

            batch_acc = (output.argmax(1) == label.view(-1,)).float().sum()
            loss_avgmeter.update(loss.detach(),data.size(0))
            acc_avgmeter.update(batch_acc.detach(),data.size(0),True)


        time_elapsed = time.time() - start_time

        train_avg_loss = loss_avgmeter.avg
        train_avg_acc = acc_avgmeter.avg

        test_avg_loss,test_avg_acc = test(model,test_loader)

        print("""Epoch:{}/{}, Avg Train Loss:{:.6f}, Avg Train Acc:{:.4f}, Avg Test Loss:{:.6f}, Avg Test Acc:{:.4f}, Best Acc:\033[91m{:.4f} \033[0m""".\
                format(epoch,args.epochs,train_avg_loss,train_avg_acc,test_avg_loss,test_avg_acc,best_acc))
        logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train_Loss: {train_avg_loss}, Train_Accuracy: {train_avg_acc},'
                     f' Test_Loss: {test_avg_loss}, Test_Accuracy: {test_avg_acc}')
        print('Elapsed Time: {:.0f}m {:.0f}s'.format(time_elapsed 
        print('Current learning rate:{:.6f}'.format(lr))

        if test_avg_loss < test_loss_min:
            test_loss_min = test_avg_loss
            counter = 0
        else:
            counter += 1

        loss_avgmeter.reset()
        acc_avgmeter.reset()


        if best_acc<=test_avg_acc:
            
            torch.save(model,args.model_dir + save_name + '.pt') 
            best_acc=test_avg_acc

        test_avg_loss, test_avg_acc = test(model, test_loader)
        print('bn acc={:.4f}, bn loss={:.4f}'.format(test_avg_acc, test_avg_loss))

        

if __name__ == '__main__':
    main()

