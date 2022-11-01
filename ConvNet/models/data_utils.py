import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict,deque
import pickle
import os
from .filter import *
from scipy.ndimage import filters

CIFAR10 = "cifar10"
MNIST = "mnist"
CIFAR100 = "cifar100"

class Cifar10(torchvision.datasets.CIFAR10):
    """cifar-10 dataset.

    load n images from every class in cifar-10 training/testing set.

    Attributes:
        path: String, path to dataset file.
        transforms: transforms
        n_images_per_class: Integer, number of image per class.
        train: Boolean, load training set(True) or testing set(False).
    """
    def __init__(self, path, transforms, train=True, n=10000):
        super().__init__(path, train, download=True)
        self.transforms = transforms
        self.n_images_per_class = n
        self.n_classes = 10
        self.new2old_indices = self.create_idx_mapping()

    def create_idx_mapping(self):
        label2idx = defaultdict(lambda: deque(maxlen=self.n_images_per_class))
        count = defaultdict(int)
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)
            count[label] += 1
        new2old = []
        for i in range(10):
            for item in label2idx[i]:
                new2old.append(item)
        return new2old

    def __len__(self):
        return len(self.new2old_indices)

    def __getitem__(self, index):
        im, label = super().__getitem__(self.new2old_indices[index])
        return self.transforms(im), label

class Mnist(torchvision.datasets.MNIST):
    """MNIST dataset.

    load n images from every class in MNIST training/testing set.

    Attributes:
        path: String, path to dataset file.
        transforms: transforms
        n_images_per_class: Integer, number of image per class.
        train: Boolean, load training set(True) or testing set(False).
    """
    def __init__(self, path, transforms, train=True, n=10000):
        super().__init__(path, train, download=True)
        self.transforms = transforms
        self.n_images_per_class = n
        self.n_classes = 10
        self.new2old_indices = self.create_idx_mapping()

    def create_idx_mapping(self):
        label2idx = defaultdict(lambda: deque(maxlen=self.n_images_per_class))
        count = defaultdict(int)
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)
            count[label] += 1
        new2old = []
        for i in range(10):
            for item in label2idx[i]:
                new2old.append(item)
        return new2old

    def __len__(self):
        return len(self.new2old_indices)

    def __getitem__(self, index):
        im, label = super().__getitem__(self.new2old_indices[index])
        return self.transforms(im), label

class FMnist(torchvision.datasets.FashionMNIST):
    """Fashion MNIST dataset(not tested yet).

    load n images from every class in Fashion MNIST training/testing set.

    Attributes:
        path: String, path to dataset file.
        transforms: transforms
        n_images_per_class: Integer, number of image per class.
        train: Boolean, load training set(True) or testing set(False).
    """
    def __init__(self, path, transforms, train=True, n=10000):
        super().__init__(path, train, download=True)
        self.transforms = transforms
        self.n_images_per_class = n
        self.n_classes = 10
        self.new2old_indices = self.create_idx_mapping()

    def create_idx_mapping(self):
        label2idx = defaultdict(lambda: deque(maxlen=self.n_images_per_class))
        count = defaultdict(int)
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)
            count[label] += 1
        new2old = []
        for i in range(10):
            for item in label2idx[i]:
                new2old.append(item)
        return new2old

    def __len__(self):
        return len(self.new2old_indices)

    def __getitem__(self, index):
        im, label = super().__getitem__(self.new2old_indices[index])
        return self.transforms(im), label


def get_data(datasetName='cifar10',
             transform=transforms.Compose(
                 [transforms.ToTensor(),
                  transforms.Normalize((0.1307,), (0.3081,))]),
             size = 10000,
             sparsity_gt = 0, # sparsity controller
             filter = "none", # choose between ['none', 'lowpass', 'highpass']
             sigma = 1.0    # Gaussian filter hyper-parameter
             ):
    """get training data and test data.

    Args:
        dataset: 'Cifar10' or 'Mnist' or 'FMnist'
        transform: transforms
        size: size of test set
    """
    if datasetName == CIFAR10:
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                       download=True, transform=transform)
        # testset = Cifar10('../data', transform, False, size)
        trainset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]))
        testset = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]))
    elif datasetName == CIFAR100:
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                       download=True, transform=transform)
        # testset = Cifar10('../data', transform, False, size)
        trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]))
        testset = torchvision.datasets.CIFAR100('./data', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]))
    elif datasetName == 'FMnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                              download=True, transform=transform)
        testset = FMnist('../data', transform, False, size)
    elif datasetName == MNIST:
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transforms.Compose([
                                         # transforms.Pad(4),
                                         # transforms.RandomCrop(32),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
        testset = Mnist('../data', transforms.Compose([
                                         # transforms.Pad(4),
                                         # transforms.RandomCrop(32),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.Lambda(lambda x:
                                                           filters.gaussian_filter(x,
                                                                sigma) if filter == 'lowpass' else x),
                                         transforms.Lambda(lambda x:
                                                           my_gaussian_filter_2(x, 1 / sigma,
                                                                filter) if filter == 'highpass' else x),
                                         transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: torch.where(x > sparsity_gt, x, torch.zeros_like(
                                                 x)) if sparsity_gt > 0 else x),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]), train=False, n=size)
    else:
        raise Exception('Wrong dataset name %s' % (datasetName))

    return trainset, testset

def get_dataset_setting(datasetName):
    """get dataset args.

    Returns:
        in_channels: interger
        size: integer, size of data pics
    """
    if datasetName == MNIST:
        return 1, 28, 10
    elif datasetName == 'FMnist':
        return 1, 28, 10
    elif datasetName == CIFAR10:
        return 3, 32, 10
    elif datasetName == CIFAR100:
        return 3, 32, 100
    elif datasetName == "imagenet":
        return 3, 224, 1000
    else:
        raise Exception('Wrong dataset name: %s' % (datasetName))

def save_pickle(data, path, fn):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path+fn, "wb")
    pickle.dump(data,f)
    f.close()

def load_pickle(path):
    f = open(path, "rb")
    b = pickle.load(f)
    f.close()
    return b

def save_model(model, modelName, datasetName, accuracy=0, path="",cfg=None):
    """save model to math.
    """
    dirpath = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    torch.save({'modelName':modelName,
                'datasetName':datasetName, 'accuracy': accuracy, 'cfg': cfg, 'state_dict': model.state_dict()}, path)

def load_model(model, path):
    """load model from path.

    Returns:
        model: model loaded from file.
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        return model, checkpoint['modelName'], checkpoint['datasetName'], checkpoint['cfg'], checkpoint['accuracy']
        # return model, checkpoint['modelName'], checkpoint['datasetName']
    else:
        print("no checkpoint found at '%s'" % (path))
    return