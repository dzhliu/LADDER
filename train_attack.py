import argparse


from models.resnet import *
from models.preact_resnet import *
from models.cnn_mnist import *
import torch.optim as optim
from data_poison import *
import logging
from smoothing import *
from utils_lfba import *
from models.googlenet import *
from models.vgg import *

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

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--model_dir', type=str, default="./saved_model/")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default="cnn_mnist",
                        help='cnn_mnist, preact_resnet, resnet18')
    parser.add_argument('--data', type=str, default="svhn",
                        help='svhn, gtsrb, cifar10, imagenet, celeba， imagenet_full')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=10)

    
    parser.add_argument('--attack_mode', type=str, default="HCB",
                        help='square, sig, refool, ftrojan, fiba, freq, patch_trigger, BELT, HCB')
    parser.add_argument('--poison_ratio', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, default=7)
    
    parser.add_argument('--smooth_type', type=str, default="no_smooth",
                        help='gaussian, wiener, BM3D, no_smooth')

    return parser.parse_args()

args = args_parser()
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)


if args.data == "imagenet_full":
    trainset = torch.load('./data/train_sub_imagenet.pt')
    testset = torch.load('./data/test_sub_imagenet.pt')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    print("a pre-selected imagenet full dataset (10 classes) has been loaded!")
else:
    
    train_dataset, test_dataset = load_dataset(args)
    clean_train_dataset = copy.deepcopy(train_dataset)
    clean_test_dataset = copy.deepcopy(test_dataset)
    train_loader, test_loader = load_data(args, clean_train_dataset, clean_test_dataset)
print('finish loading dataset')

def test(model, test_loader):

    model.eval()
    clean_loss_avgmeter = AverageMeter()
    clean_acc_avgmeter = AverageMeter()
    bd_loss_avgmeter = AverageMeter()
    bd_acc_avgmeter = AverageMeter()

    for batch_idx, (data, label) in enumerate(test_loader):
        
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc=(output.argmax(1) == label.view(-1,)).float().sum()

        loss=criterion(output,label.view(-1,))
        clean_loss_avgmeter.update(loss.detach(),data.size(0))
        clean_acc_avgmeter.update(batch_acc.detach(),data.size(0),True)

    for batch_idx, (data, label) in enumerate(test_loader):
        if args.attack_mode == 'square':
            data, label = square_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'sig':
            data, label = sig_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'blend':
            data, label = blend_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'ftrojan':
            data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'fiba':
            data, label = fiba_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'patch_trigger':
            data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'BELT':
            data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=1.0)
        elif args.attack_mode == 'HCB':
            data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=1.0)
        else:
            raise Exception(f'Error, unknown attack mode{args.attack_mode}')
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()

        loss = criterion(output, label.view(-1, ))
        bd_loss_avgmeter.update(loss.detach(), data.size(0))
        bd_acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

    model.train()

    return clean_loss_avgmeter.avg, clean_acc_avgmeter.avg, bd_loss_avgmeter.avg, bd_acc_avgmeter.avg

def main():
    if args.data == "cifar10":
        args.num_classes = 10
    elif args.data == "gtsrb":
        args.num_classes = 43
    elif args.data == "imagenet":
        args.num_classes = 200
    elif args.data == "celeba":
        args.num_classes = 8
    elif args.data == "mnist":
        args.num_classes = 10
    elif args.data == "svhn":
        args.num_classes = 10
    elif args.data == "imagenet_full":
        args.num_classes = 1000

    if args.data == "imagenet_full":
        if args.model == "resnet":
            print('load torchvision.models.resnet18 with IMAGENET1K_V1 pretrained weights on imagenet task')
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            
        else:
            raise Exception('error, the model '+args.model+' is not supported for imagenet_full dataset')
    else:
        if args.model == "vgg16":
            model = vgg16_bn(num_classes=args.num_classes)
        elif args.model == "vgg11":
            model = vgg11_bn(num_classes=args.num_classes)
        elif args.model == "preact_resnet":
            model = PreActResNet18(num_classes=args.num_classes)
        elif args.model == "cnn_mnist":
            model = CNN_MNIST()
        elif args.model == "resnet":
            
            
            from models.resnet_cifar10 import ResNet18
            model = ResNet18(num_classes=args.num_classes)
        elif args.model == "googlenet":
            model = GoogLeNet()
        else:
            raise Exception('unknown model name')

    model = model.to(args.device)

    save_name = "train_attack_" + args.attack_mode + "_" + args.model + "_" + args.data + "_" + str(args.poison_ratio)
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
    
    for epoch in range(args.epochs):

        model.train()

        if counter / 10 == 1:
            counter = 0
            lr = lr * 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        for batch_idx, (data, label) in enumerate(train_loader):

            
            
            
            
            
            
            

            if args.attack_mode == 'square':
                data, label = square_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'sig':
                data, label = sig_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'blend':
                data, label = blend_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'ftrojan':
                data, label = ftrojan_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'fiba':
                data, label = fiba_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'patch_trigger':
                data, label = patch_trigger(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'BELT':
                data, label = BELT_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            elif args.attack_mode == 'HCB':
                data, label = HCB_poison(args, data, label, args.target_label, poison_ratio=args.poison_ratio)
            else:
                raise Exception(f'Error, unknown attack mode{args.attack_mode}')
            data = data.to(args.device)
            label = label.to(args.device)
            output = model(data)
            batch_acc = (output.argmax(1) == label.view(-1, )).float().sum()
            optimizer.zero_grad()
            loss = criterion(output,label.view(-1,))
            loss_avgmeter.update(loss.detach(), data.size(0))
            acc_avgmeter.update(batch_acc.detach(), data.size(0), True)

            loss.backward()
            optimizer.step()

        time_elapsed = time.time() - start_time

        train_avg_loss = loss_avgmeter.avg
        train_avg_acc = acc_avgmeter.avg

        clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss, bd_test_avg_acc = test(model, test_loader)

        print("""{}/{}, Avg Train Loss:{:.6f}, Avg Train Acc:{:.4f}, 
        Avg Test Loss Clean:{:.6f}, Avg Test Acc Clean:{:.4f}, 
        Avg Test Loss backdoor:{:.6f}, Avg Test Acc backdoor:{:.4f}, Best Acc:\033[91m{:.4f} \033[0m,
        Best Backdoor Acc:\033[91m{:.4f} \033[0m""".\
                format(epoch,args.epochs, train_avg_loss, train_avg_acc,
                       clean_test_avg_loss, clean_test_avg_acc, bd_test_avg_loss,
                       bd_test_avg_acc,best_acc,best_bd_acc))
        logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train_Loss: {train_avg_loss}, Train_Accuracy: {train_avg_acc}, '
                     f'Clean_Test_Loss: {clean_test_avg_loss}, Clean_Test_Accuracy: {clean_test_avg_acc}, '
                     f'Bd_Test_Loss: {bd_test_avg_loss}, Bd_Test_Accuracy: {bd_test_avg_acc}')
        print('Elapsed Time: {:.0f}m {:.0f}s'.format(time_elapsed 
        print('Current learning rate:{:.6f}'.format(lr))
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
