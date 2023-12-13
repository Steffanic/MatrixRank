
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


def get_train_test_dataloaders(dataset:str):
    match dataset:
        case 'mnist':
            train_dataset = dsets.MNIST(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            test_dataset = dsets.MNIST(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor())

        case 'cifar10':
            train_dataset = dsets.CIFAR10(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            test_dataset = dsets.CIFAR10(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor())

        case 'cifar100':
            train_dataset = dsets.CIFAR100(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

            test_dataset = dsets.CIFAR100(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor())
            
        case _:
            raise ValueError(f'Unknown dataset: {dataset}')
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    
    return train_loader, test_loader

def get_model(model:str, num_classes:int):
    match model:
        case 'mlp':
            model = MLP([int(x) for x in args.model_spec.strip("[]").split(",")])
        case 'cnn':
            model = CNN()
        case _:
            raise ValueError(f'Unknown model: {model}')
    
    return model

def MLP(model_spec:list):
    layers = []
    for i in range(len(model_spec)-1):
        layers.append(nn.Linear(model_spec[i], model_spec[i+1]))
        if i != len(model_spec)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def CNN():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7*7*64, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )

def get_loss(loss:str):
    match loss:
        case 'cross_entropy':
            return nn.CrossEntropyLoss()
        case 'mse':
            return nn.MSELoss()
        case _:
            raise ValueError(f'Unknown loss: {loss}')
        
def get_matrix_ranks(model):
    ranks = []
    for param in model.parameters():
        if param.data.ndim == 2:
            ranks.append(torch.linalg.matrix_rank(param.data))
    return ranks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs to train before reparameterizing the network')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use[mnist, cifar10, cifar100]')
    parser.add_argument('--model', type=str, default='mlp', help='model to use[mlp, cnn]')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--model_spec', type=str, default="[784, 100, 100, 100, 10]", help='model specification', required=False)
    parser.add_argument('--num_meta_train', type=int, default=10, help='number of reparameterizing steps')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function to use[cross_entropy, mse]')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')

    args = parser.parse_args()

    train_dataloader, test_dataloader = get_train_test_dataloaders(args.dataset)

    initial_model = get_model(args.model, args.num_classes)

    loss_fn = get_loss(args.loss)



    optimized_ranks = []

    for i in range(args.num_meta_train):
        if i!=0:
            model = get_model(optimized_ranks, args.num_classes)
        else:
            model = initial_model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        for epoch in range(args.max_epochs):
            for i, (images, labels) in enumerate(train_dataloader):
                if args.model == "mlp":
                    images = Variable(images.view(-1, 28*28))
                else:
                    images = Variable(images)
                labels = Variable(labels)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    optimized_ranks =get_matrix_ranks(model)
                    print(f'Epoch [{epoch+1}/{args.max_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Rank: {optimized_ranks}')

            lr_scheduler.step(loss)

 

