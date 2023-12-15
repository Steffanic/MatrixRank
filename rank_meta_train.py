
import datetime
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


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
                                                batch_size=len(test_dataset),
                                                shuffle=False)
    
    return train_loader, test_loader

def get_model(model:str, model_spec:list):
    match model:
        case 'mlp':
            model = MLP([int(x) for x in model_spec])
        case 'cnn':
            model = CNN()
        case _:
            raise ValueError(f'Unknown model: {model}')
    
    return model

def MLP(model_spec:list):
    layers = []
    input_size = 784 if args.dataset == 'mnist' else 1024
    num_classes = 100 if args.dataset == 'cifar100' else 10
    layers.append(nn.Linear(input_size, model_spec[0]))
    for i in range(len(model_spec)-1):
        layers.append(nn.Linear(model_spec[i], model_spec[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_spec[-1], num_classes))
    return nn.Sequential(*layers)

def CNN():
    num_classes = 100 if args.dataset == 'cifar100' else 10
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
        nn.Linear(1024, num_classes)
    )

def get_loss(loss:str):
    match loss:
        case 'cross_entropy':
            return nn.CrossEntropyLoss(reduction='mean')
        case 'mse':
            return nn.MSELoss()
        case _:
            raise ValueError(f'Unknown loss: {loss}')
        
def get_matrix_ranks(model):
    ranks = []
    for param in model.parameters():
        if param.data.ndim == 2:
            ranks.append(torch.linalg.matrix_rank(param.data).cpu())
        elif param.data.ndim == 4:
            # here we are going to compute the rank of each (kxk) filter. the output should be a [out_channels, in_channels, rank] Tensor
            # we will do this by iterating through each out_channel and in_channel and computing the rank of the matrix formed by the (kxk) filters
            
            # first we need to get the shape of the filter
            k = param.data.shape[2]
            # next we need to get the number of filters
            out_channels = param.data.shape[0]
            in_channels = param.data.shape[1]
            # now we need to iterate through each filter and compute the rank
            filter_ranks = []
            for i in range(out_channels):
                for j in range(in_channels):
                    filter_ranks.append(torch.linalg.matrix_rank(param.data[i, j, :, :]).cpu())
            ranks.append(torch.Tensor(filter_ranks).reshape(out_channels, in_channels))
        else:
            pass
        
    return ranks

def get_test_accuracy(model, test_dataloader):
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        if args.model == "mlp":
            images = Variable(images.view(-1, 28*28))
        else:
            images = Variable(images)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    return 100 * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs to train before reparameterizing the network')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use[mnist, cifar10, cifar100]')
    parser.add_argument('--model', type=str, default='mlp', help='model to use[mlp, cnn]')
    parser.add_argument('--model_spec', type=str, default="[1000, 1000, 1000, 10]", help='model specification', required=False)
    parser.add_argument('--num_meta_train', type=int, default=10, help='number of reparameterizing steps')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function to use[cross_entropy, mse]')
    parser.add_argument('--weight_decay', type=float, default=0.00000001, help='weight decay')
    parser.add_argument('--results_path', type=str, default='C:/Users/pat/Documents/rank/MatrixRank/results', help='path to save results')

    args = parser.parse_args()

    train_dataloader, test_dataloader = get_train_test_dataloaders(args.dataset)

    initial_model = get_model(args.model, [x for x in args.model_spec.strip("[]").split(",")])

    loss_fn = get_loss(args.loss)

    test_rate = len(train_dataloader)//30 or 1
    rank_histories = {}
    training_loss_history = {}
    training_accuracy_history = {}
    test_accuracy_history = {}
    optimized_ranks = []
    meta_progress = tqdm(range(args.num_meta_train), desc="Meta Training", position=0, leave=True)
    epoch_progress = tqdm(range(args.max_epochs), desc="Epoch", position=1, leave=True)
    step_progress = tqdm(total=len(train_dataloader), desc="Step", position=2, leave=True)

    for meta_step in range(args.num_meta_train):
        if meta_step!=0:
            if args.model == "mlp":
                model = get_model(args.model, optimized_ranks)
            elif args.model == "cnn":
                print("CNN reparameterization not yet implemented. Continuing with initial model. printing ranks, though.")
        else:
            model = initial_model

        if torch.cuda.is_available():  
            model.cuda()
            loss_fn.cuda()
        
        rank_histories[meta_step] = []
        training_loss_history[meta_step] = []
        training_accuracy_history[meta_step] = []
        test_accuracy_history[meta_step] = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        for epoch in range(args.max_epochs):
            for i, (images, labels) in enumerate(train_dataloader):
                if args.model == "mlp":
                    images = Variable(images.view(-1, 28*28))
                else:
                    images = Variable(images)
                labels = Variable(labels)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i+1) % test_rate == 0:
                    optimized_ranks =get_matrix_ranks(model)
                    rank_histories[meta_step].append(optimized_ranks)
                    training_loss_history[meta_step].append(loss.mean().item())
                    training_accuracy_history[meta_step].append(get_test_accuracy(model, train_dataloader))
                    test_accuracy_history[meta_step].append(get_test_accuracy(model, test_dataloader))
                    test_acc = get_test_accuracy(model, test_dataloader)
                    step_progress.set_description(f'Step [{i+1}/{len(train_dataloader)}], Loss: {loss.mean().item():.3f}, Test Accuracy: {test_acc:.3f}')
            
                step_progress.update(1)
            step_progress.reset()
            epoch_progress.update(1)
            epoch_progress.set_description(f'Epoch [{epoch+1}/{args.max_epochs}], Loss: {loss.mean().item():.4f}, Test Accuracy: {test_acc:.3f}')
            evolution_threshold = 95 if meta_step <= 5 else 98
            if test_acc > evolution_threshold:
                epoch_progress.reset()
                break

            lr_scheduler.step(loss.mean().item())
        meta_progress.update(1)
        meta_progress.set_description(f'Meta Step [{meta_step+1}/{args.num_meta_train}], Loss: {loss.mean().item():.4f}, Test Accuracy: {test_acc:.3f}, Rank: {[int(x) for x  in optimized_ranks]}')

    print(f'Final Rank: {optimized_ranks}')
    print(f'Final Test Accuracy: {test_acc}')
    print(f'Final Loss: {loss.mean().item():.4f}')
    

    filename_prefix = f'{args.dataset}_{args.model}_{args.batch_size}_{args.weight_decay}_{len(args.model_spec)}'
    # plot the rank evolution
    fig, ax = plt.subplots()
    for meta_step in range(args.num_meta_train):
        ax.plot(training_accuracy_history[meta_step], label=f'Meta Step {meta_step+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_training_accuracy.png')
    plt.show()

    fig, ax = plt.subplots()
    for meta_step in range(args.num_meta_train):
        ax.plot(test_accuracy_history[meta_step], label=f'Meta Step {meta_step+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_test_accuracy.png')
    plt.show()

    fig, ax = plt.subplots()
    for meta_step in range(args.num_meta_train):
        ax.plot(training_loss_history[meta_step], label=f'Meta Step {meta_step+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_training_loss.png')
    plt.show()

    if args.model == "mlp":
        average_ranks = [[torch.mean(x) for x in rank_history] for rank_history in rank_histories.values()]
        maximum_ranks = [[torch.max(x) for x in rank_history] for rank_history in rank_histories.values()]
        rank_series = [[[[x[rank_ind] for x in epoch] for epoch in meta_step] for meta_step in rank_histories.values()] for rank_ind in range(len(rank_histories[0][0][0]))]
    elif args.model == "cnn":
        # the difficulty here is handling the fact that the ranks are now a list of tensors, not a tensor due to the fact that we are computing the rank of each filter
        average_ranks = [[[ranks if len(ranks)==1 else torch.mean(ranks) for ranks in epoch_step] for epoch_step in rank_history] for rank_history in rank_histories.values()]
        average_ranks = [[torch.mean(x) for x in rank_history] for rank_history in average_ranks]
        maximum_ranks = [[[ranks if len(ranks)==1 else torch.max(ranks) for ranks in epoch_step] for epoch_step in rank_history] for rank_history in rank_histories.values()]
        maximum_ranks = [[torch.max(x) for x in rank_history] for rank_history in maximum_ranks]
        linear_rank_series = [[[[x[rank_ind] for x in epoch if len(x[rank_ind])==1] for epoch in meta_step] for meta_step in rank_histories.values()] for rank_ind in range(len(rank_histories[0][0][0]))]
        filter_rank_series = [[[[[[x[rank_ind][in_channel][out_channel] for x in epoch if len(x[rank_ind])!=1] for epoch in meta_step] for meta_step in rank_histories.values()] for in_channel in range(len(rank_histories[0][0][0]))] for out_channel in range(len(rank_histories[0][0][0]))] for rank_ind in range(len(rank_histories[0][0][0]))]



    fig, ax = plt.subplots()
    for meta_step in range(args.num_meta_train):
        ax.plot(average_ranks[meta_step], label=f'Meta Step {meta_step+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Rank')
    ax.legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_average_rank.png')
    plt.show()

    fig, ax = plt.subplots()
    for meta_step in range(args.num_meta_train):
        ax.plot(maximum_ranks[meta_step], label=f'Meta Step {meta_step+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Maximum Rank')
    ax.legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_maximum_rank.png')
    plt.show()

    if args.model == "mlp":
        ranks = rank_series
    elif args.model == "cnn":
        ranks = linear_rank_series
    fig, ax = plt.subplots((args.num_meta_train+1)//2, 2, figsize=(10, 10))
    for meta_step in range(args.num_meta_train):
        for rank in ranks:
            ax[meta_step//2, meta_step%2].plot(rank[meta_step], label=f'Layer {i+1}')
        ax[meta_step//2, meta_step%2].set_xlabel('Epoch')
        ax[meta_step//2, meta_step%2].set_ylabel('Ranks')
        ax[meta_step//2, meta_step%2].legend()
    plt.savefig(f'{args.results_path}/{filename_prefix}_all_ranks.png')
    plt.show()

    if args.model == "cnn":
        fig, ax = plt.subplots((args.num_meta_train+1)//2, 2, figsize=(10, 10))
        for meta_step in range(args.num_meta_train):
            for in_channel in range(len(rank_histories[0][0][0])):
                for out_channel in range(len(rank_histories[0][0][0])):
                    ax[meta_step//2, meta_step%2].plot(filter_rank_series[meta_step][in_channel][out_channel], label=f'({in_channel}, {out_channel})')
            ax[meta_step//2, meta_step%2].set_xlabel('Epoch')
            ax[meta_step//2, meta_step%2].set_ylabel('Ranks')
            ax[meta_step//2, meta_step%2].legend()
        plt.savefig(f'{args.results_path}/{filename_prefix}_filter_ranks.png')
        plt.show()

