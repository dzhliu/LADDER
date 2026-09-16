


import argparse
import sys
import math
import torch

import utils_lfba
from models.resnet import *
from models.preact_resnet import *
from models.cnn_mnist import *
import torch.optim as optim
import logging
from smooth import *
from utils_lfba import *

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--model_dir', type=str, default="./saved_model/")
    parser.add_argument('--device', type=str, default="cuda:0")  
    parser.add_argument('--data', type=str, default="cifar10",
                        help='svhn, gtsrb, cifar10, imagenet, celeba')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='preact_resnet',
                        help="cnn_mnist, vgg11, vgg16, preact_resnet, resnet, googlenet")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--attack_mode', type=str, default="base", help='base')
    parser.add_argument('--poison_ratio', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--epochs_t', type=int, default=3)
    parser.add_argument('--epochs_model', type=int, default=3)

    parser.add_argument('--alpha', type=float, default=0.95)

    return parser.parse_args()

args = args_parser()
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)


train_dataset, test_dataset = load_dataset(args)
train_loader, test_loader = load_data(args, train_dataset, test_dataset)
print('finish loading dataset')

def test(model, test_loader, t):

    model.eval()
    clean_loss_avgmeter = AverageMeter()
    clean_acc_avgmeter = AverageMeter()
    bd_loss_avgmeter = AverageMeter()
    bd_acc_avgmeter = AverageMeter()

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()

        loss = criterion(output, label.view(-1, ))
        clean_loss_avgmeter.update(loss.detach(), data.size(0))
        clean_acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data_poison(data, label, args.target_label, poison_ratio=1.0, t=t)
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()

        loss = criterion(output, label.view(-1, ))
        bd_loss_avgmeter.update(loss.detach(), data.size(0))
        bd_acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

    model.train()

    return clean_loss_avgmeter.avg, clean_acc_avgmeter.avg, bd_loss_avgmeter.avg, bd_acc_avgmeter.avg

def data_poison_frac(data,label,target_label,poison_ratio=0.1, t=None):
    batch_size, c, h, w = data.shape
    poison_num = math.ceil(batch_size * poison_ratio)
    t_copy = copy.deepcopy(t)
    for idx in range(poison_num):
        label[idx] = target_label
        data_freq = data[idx]
        data_freq = data_freq.numpy()
        data_freq = utils_lfba.DCT(data_freq, h, False)
        data_freq = data_freq + t_copy.detach().numpy()
        data_freq = utils_lfba.IDCT(data_freq, h, False)
        data[idx] = torch.clamp(torch.tensor(data_freq), 0, 1)

    return data, label

def data_poison(data,label,target_label,poison_ratio=0.1, t=None):

    
    
    
    
    
    

    data, label = data_poison_frac(data, label, target_label, poison_ratio, t)
    return data, label

def main():

    if args.data == "cifar10":
        args.num_classes = 10
        t = torch.randn((3, 32, 32), requires_grad=True)
        t = t/t.max()
        t = torch.tensor(t,requires_grad=True)
    elif args.data == "gtsrb":
        args.num_classes = 43
        t = torch.randn((3, 32, 32), requires_grad=True)
    elif args.data == "imagenet":
        args.num_classes = 200
        t = torch.randn((3, 64, 64), requires_grad=True)
    elif args.data == "celeba":
        args.num_classes = 8
        t = torch.randn((3, 64, 64), requires_grad=True)
    elif args.data == "svhn":
        args.num_classes = 10
        t = torch.randn((3, 32, 32), requires_grad=True)


    if args.model == "preact_resnet":
        model = PreActResNet18(num_classes=args.num_classes)
    elif args.model == "cnn_mnist":
        model = CNN_MNIST()
    elif args.model == "resnet":
        model = ResNet18(num_classes=args.num_classes)
    else:
        raise Exception('unknown model name')
    model = torch.load('./saved_model/train_clean_preact_resnet_cifar10.pt')
    model = model.to(args.device)

    save_name = "train_conflict_attack_" + args.attack_mode + "_" + args.model + "_" + args.data + "_" + str(
        args.poison_ratio)
    logging.basicConfig(filename=args.log_dir + save_name + '.txt', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.FileHandler(args.log_dir + save_name + '.txt', mode='w+')

    loss_avgmeter = AverageMeter()
    acc_avgmeter = AverageMeter()


    counter = 0
    best_acc = 0
    best_bd_acc = 0
    clean_test_loss_min = 0
    lr = args.lr
    start_time = time.time()

    
    for total_i in range(50):
        for epoch in range(args.epochs_t):
            model.train()

            if counter / 10 == 1:
                counter = 0
                lr = lr * 0.5

            optimizer = optim.SGD([t], lr=lr)

            for batch_idx, (data, label) in enumerate(train_loader):
                l2_norm = torch.norm(t, p=2)
                train_data, train_label = data_poison(data, label, args.target_label, poison_ratio=args.poison_ratio, t=t)
                bd_data, bd_label = data_poison(data, label, args.target_label, poison_ratio=1.0, t=t)
                train_data = train_data.to(args.device)
                train_label = train_label.to(args.device)
                output = model(train_data)
                batch_acc = (output.argmax(1) == train_label.view(-1, )).float().sum()
                optimizer.zero_grad()
                loss = args.alpha * criterion(output, train_label.view(-1, )) + (1- args.alpha) * l2_norm
                loss_avgmeter.update(loss.detach(), train_data.size(0))
                acc_avgmeter.update(batch_acc.detach(), train_data.size(0), True)

                loss.backward()
                optimizer.step()
                
            print('l2_norm:',l2_norm, flush=True)
        

        for epoch in range(args.epochs_model):

            model.train()

            if counter / 10 == 1:
                counter = 0
                lr = lr * 0.5
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            for batch_idx, (data, label) in enumerate(train_loader):
                l2_norm = torch.norm(t, p=2)
                data, label = data_poison(data, label, args.target_label, poison_ratio=args.poison_ratio, t=t)
                data = data.to(args.device)
                label = label.to(args.device)
                output = model(data)
                batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()
                optimizer.zero_grad()
                loss = args.alpha * criterion(output, label.view(-1, )) + (1- args.alpha) * l2_norm
                loss_avgmeter.update(loss.detach(), data.size(0))
                acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

                loss.backward()
                optimizer.step()

            time_elapsed = time.time() - start_time

            train_avg_loss = loss_avgmeter.avg
            train_avg_acc = acc_avgmeter.avg

            clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss, bd_test_avg_acc = test(model, test_loader, t)

            print("""{}/{}, Avg Train Loss:{:.6f}, Avg Train Acc:{:.4f}, 
            Avg Test Loss Clean:{:.6f}, Avg Test Acc Clean:{:.4f}, 
            Avg Test Loss Backdoor:{:.6f}, Avg Test Acc Backdoor:{:.4f}, Best Acc:\033[91m{:.4f} \033[0m, 
            Best Backdoor Acc:\033[91m{:.4f} \033[0m""". \
                  format(epoch, args.epochs_model, train_avg_loss, train_avg_acc,
                         clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss,
                         bd_test_avg_acc, best_acc, best_bd_acc))
            logging.info(
                f'Epoch {epoch + 1}/{args.epochs_model}, Train_Loss: {train_avg_loss}, Train_Accuracy: {train_avg_acc}, '
                f'Clean_Test_Loss: {clean_test_avg_loss}, Clean_Test_Accuracy: {clean_test_avg_acc}, '
                f'Bd_Test_Loss: {bd_test_avg_loss}, Bd_Test_Accuracy: {bd_test_avg_acc}')
            print('Elapsed Time: {:.0f}m {:.0f}s'.format(time_elapsed 
            print('Current learning rate:{:.6f}'.format(lr), flush=True)
            loss_avgmeter.reset()
            acc_avgmeter.reset()

            if best_bd_acc <= bd_test_avg_acc:
                best_bd_acc = bd_test_avg_acc

            if best_acc <= clean_test_avg_acc:
                best_acc = clean_test_avg_acc

            if clean_test_avg_loss < clean_test_loss_min:
                clean_test_loss_min = clean_test_avg_loss
                counter = 0
            else:
                counter += 1

            
            path = os.path.join(args.model_dir, f'{save_name}.pt')
            torch.save(model, path)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
