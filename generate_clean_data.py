print('Load modules...')
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
from models.vgg_mnist import VGG  # Use the MNIST version of VGG
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net", default='mnist', help="the network used for the attack, either mnist, cif10 or cif100")
args = parser.parse_args()
# choose attack
net = args.net

print('Load model...')

if net == 'mnist':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_mnist.pth')  # Ensure you have the trained model saved here
    model.load_state_dict(checkpoint['net'])
    
    # normalizing the data
    print('Load MNIST test set')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    testset_normalized = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

elif net == 'cif10':
    from models.vgg_cif10 import VGG  # Ensure you have the correct CIFAR-10 VGG model definition
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    # normalizing the data
    print('Load CIFAR-10 test set')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset_normalized = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

elif net == 'cif100':
    from models.vgg import vgg16_bn  # Ensure you have the correct CIFAR-100 VGG model definition
    model = vgg16_bn()
    checkpoint = torch.load('./models/vgg_cif100.pth')
    model.load_state_dict(checkpoint['net'])
    
    # normalizing the data
    print('Load CIFAR-100 test set')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    testset_normalized = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

else:
    print('unknown net')
    
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_iter = iter(testloader)
clean_dataset = []
correct = 0
total = 0
i = 0
print('Classify images...')
for images, labels in testloader_normalized:
    data = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    if (predicted == labels):
        clean_dataset.append(data)
    i += 1

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

torch.save(clean_dataset, './data/clean_data_'+net)
print('Done extracting and saving correctly classified images!')
