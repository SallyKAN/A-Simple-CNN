import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import time as time
from torchvision.datasets import ImageFolder
import os


def train(mode,device,sheduler,epoches,train_loader,val_loader,testloader):
    train_losses = []
    val_losses = []
    test_accuracies = []
    val_accuracies = []
    t0 = time.time()
    for epoch in range(epoches):  # loop over the dataset multiple times
        sheduler.step()
        train_running_loss = 0.0
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print statistics
#             losses.append(running_loss)
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 losses.append(running_loss / 2000)
#                 running_loss = 0.0
#             running_loss /= (i+1)
        train_loss = train_running_loss/len(train_loader)
        train_losses.append(train_loss)
        val_running_loss = 0.0
        for i,data in enumerate(val_loader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
        val_loss = val_running_loss/len(val_loader)
        val_losses.append(val_loss)
        print('%dth epoch, train loss: %.3f, validation loss:%.3f' %(epoch + 1,train_loss,val_loss))
        val_accuracy = eval_net(net, val_loader)
        test_accuracy = eval_net(net, testloader)
        test_accuracies.append(test_accuracy)
        val_accuracies.append(val_accuracy)
        if mode == 'baseline':
            torch.save(net.state_dict(), 'baseline.pth')
        elif mode == 'modified':
            torch.save(net.state_dict(), 'modified.pth')
    print('Finished Training，take %.3f mins'% ((time.time() - t0)/60))
    print('test Acc: {:4f}'.format(test_accuracy))
    return train_losses, val_losses,test_accuracies,val_accuracies


def muti_plot_loss_and_accur(lr, epoches, train_losses_dic, val_losses_dic, test_accuracies_dic, val_accuracies_dic):
    t = np.arange(1, epoches + 1)
    train_losses_batch_2 = np.asarray(train_losses_dic[2])
    val_losses_batch_2 = np.asarray(val_losses_dic[2])
    test_accuracies_batch_2 = np.asarray(test_accuracies_dic[2])
    val_accuracies_batch_2 = np.asarray(val_accuracies_dic[2])
    train_losses_batch_4 = np.asarray(train_losses_dic[4])
    val_losses_batch_4 = np.asarray(val_losses_dic[4])
    test_accuracies_batch_4 = np.asarray(test_accuracies_dic[4])
    val_accuracies_batch_4 = np.asarray(val_accuracies_dic[4])
    train_losses_batch_8 = np.asarray(train_losses_dic[8])
    val_losses_batch_8 = np.asarray(val_losses_dic[8])
    test_accuracies_batch_8 = np.asarray(test_accuracies_dic[8])
    val_accuracies_batch_8 = np.asarray(val_accuracies_dic[8])

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(17, 10)

    ax1.set_title('training loss with bacth size=2,learning rate=%.5f' % (lr))
    ax1.set_ylabel('train loss')
    ax1.set_xlabel('Epoch')
    ax1.plot(t, train_losses_batch_2, color='b', marker='o')
    ax1.plot(t, val_losses_batch_2, color='r', marker='o')
    ax1.legend(['train', 'val'])

    ax2.set_title('training loss with bacth size=4,learning rate=%.5f' % (lr))
    ax2.set_ylabel('train loss')
    ax2.set_xlabel('Epoch')
    ax2.plot(t, train_losses_batch_4, color='b', marker='o')
    ax2.plot(t, val_losses_batch_4, color='r', marker='o')
    ax2.legend(['train', 'val'])

    ax3.set_title('training loss with bacth size=8,learning rate=%.5f' % (lr))
    ax3.set_ylabel('train loss')
    ax3.set_xlabel('Epoch')
    ax3.plot(t, train_losses_batch_8, color='b', marker='o')
    ax3.plot(t, val_losses_batch_8, color='r', marker='o')
    ax3.legend(['train', 'val'])

    ax4.set_title('training accuracy with bacth size=2,learning rate=%.5f' % (lr))
    ax4.set_ylabel('accuracy (%)')
    ax4.set_xlabel('Epoch')
    ax4.plot(t, test_accuracies_batch_2, color='g', marker='o')
    ax4.plot(t, val_accuracies_batch_2, color='r', marker='o')
    ax4.legend(['validation', 'test'])

    ax5.set_title('training accuracy with bacth size=4,learning rate=%.5f' % (lr))
    ax5.set_ylabel('accuracy (%)')
    ax5.set_xlabel('Epoch')
    ax5.plot(t, test_accuracies_batch_4, color='g', marker='o')
    ax5.plot(t, val_accuracies_batch_4, color='r', marker='o')
    ax5.legend(['validation', 'test'])

    ax6.set_title('training accuracy with bacth size=8,learning rate=%.5f' % (lr))
    ax6.set_ylabel('accuracy (%)')
    ax6.set_xlabel('Epoch')
    ax6.plot(t, test_accuracies_batch_8, color='g', marker='o')
    ax6.plot(t, val_accuracies_batch_8, color='r', marker='o')
    ax6.legend(['validation', 'test'])

    plt.tight_layout(pad=0.5, w_pad=1, h_pad=1)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def eval_net(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#     print('Accuracy: %d %%' % (100 * correct / total))
    return correct / total


def get_dataloders(batch_size):
    # transform = transforms.ToTensor()
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=train_transform)
    trainset = ImageFolder('PyTorch/release/train', transform=train_transform)
    valid_dataset = ImageFolder('PyTorch/release/train', transform=train_transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=test_transform)
    testset = ImageFolder('PyTorch/release/val', transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    data_indexs = list(range(len(trainset)))
    valid_size = 200
    np.random.shuffle(data_indexs)
    train_index, val_index = data_indexs[valid_size:], data_indexs[:valid_size]

    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(val_index)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=2)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=2)

    return train_loader, valid_loader, testloader
def plot_loss_and_accur(batch_size, lr, epoches,train_losses,val_losses,test_accuracies,val_accuracies):
    t = np.arange(1,epoches+1)
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    test_accuracies = np.asarray(test_accuracies)
    val_accuracies = np.asarray(val_accuracies)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(13,6.4)

    ax1.set_title('training loss with bacth size=%d,learning rate=%.5f' % (batch_size,lr))
    ax1.set_ylabel('train loss')
    ax1.set_xlabel('Epoch')
    ax1.plot(t, train_losses, color='b', marker='o')
    ax1.plot(t, val_losses, color='r', marker='o')
    ax1.legend(['train', 'val'])

    ax2.set_title('training accuracy with bacth size=%d,learning rate=%.5f' % (batch_size,lr))
    ax2.set_ylabel('train loss')
    ax2.set_xlabel('Epoch')
    ax2.plot(t, test_accuracies, color='g', marker='o')
    ax2.plot(t, val_accuracies, color='r', marker='o')
    ax2.legend(['validation', 'test'])

    plt.show()
    plt.tight_layout(pad=0.5, w_pad=1, h_pad=1)


if __name__ == '__main__':
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    lr = 0.001
    batch_size = 8
    epoches = 10
    train_losses_dic = {}
    val_losses_dic = {}
    test_accuracies_dic = {}
    val_accuracies_dic = {}
    # Data
    print('==> Preparing data..')
    # resize = transforms.Resize((48,48))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose(
                [
                    # resize,
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    normalize,
                 ])
    val_transform = transforms.Compose(
                [
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                 ])
    train_loader, val_loader, testloader = get_dataloders(batch_size)
    print(train_loader.batch_size)
    # Model
    print('==> Building model..')
    # net = Net()
    net = torchvision.models.resnet18(pretrained=True)
    # net = ModNet()
    mode = 'modified'
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=12, gamma=1)
    # scheduler = StepLR(optimizer, step_size=12, gamma=0.8)
    train_losses, val_losses, test_accuracies, val_accuracies = train(mode,device,scheduler,
                                                                      epoches, train_loader,
                                                                      val_loader, testloader)

    plot_loss_and_accur(batch_size, lr, epoches, train_losses,val_losses,test_accuracies,val_accuracies)

