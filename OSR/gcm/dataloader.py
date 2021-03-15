import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import cv2
from PIL import Image

# Transformations
RC   = transforms.RandomCrop(32, padding=4)
RHF  = transforms.RandomHorizontalFlip()
RVF  = transforms.RandomVerticalFlip()
NRM  = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT   = transforms.ToTensor()
TPIL = transforms.ToPILImage()


class MNIST_Dataset(Dataset):
    def __init__(self):
        self.trainset = MNIST(root='./data', train=True, download=True)
        self.testset = MNIST(root='./data', train=False, download=True)
        self.classDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                     '9': 9}
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        seen_classes = random.sample(range(0, 10), 7)
        unseen_classes = [idx for idx in range(10) if idx not in seen_classes]

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset(self.trainset, self.testset,
                                                                      seen_classes, unseen_classes, self.transform, args)

        return osr_trainset, osr_valset, osr_testset


class CIFAR10_Dataset(Dataset):
    def __init__(self):
        self.trainset = CIFAR10(root='./data', train=True, download=True)
        self.testset = CIFAR10(root='./data', train=False, download=True)
        self.classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                     'truck': 9}

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        seen_classes = random.sample(range(0, 10), 7)
        # seen_classes = range(0, 10)
        unseen_classes = [idx for idx in range(10) if idx not in seen_classes]

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_aug(self.trainset, self.testset, seen_classes,
                                                                      unseen_classes, self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset


class CIFAR100_Dataset(Dataset):
    def __init__(self):
        self.trainset = CIFAR100(root='./data', train=True, download=True)
        self.testset = CIFAR100(root='./data', train=False, download=True)

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                 [0.2673, 0.2564, 0.2762]),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        seen_classes = random.sample(range(0, 100), 50)
        # seen_classes = range(0, 10)
        unseen_classes = [idx for idx in range(100) if idx not in seen_classes]
        unseen_classes_num = args.unseen_num
        unseen_classes = random.sample(unseen_classes, unseen_classes_num)

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_aug(self.trainset, self.testset, seen_classes,
                                                                      unseen_classes, self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset


class CIFARAdd10_Dataset(Dataset):
    def __init__(self):
        self.trainset = CIFAR10(root='./data', train=True, download=True)
        self.testset = CIFAR10(root='./data', train=False, download=True)
        self.unknownset = CIFAR100(root='./data', train=False, download=True)
        self.classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                     'truck': 9}

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        # 4 non animal
        seen_classes = [0, 1, 8, 9]
        # animal in CIFAR100
        animal_classes = [1,2,3,4,6,7,11,14,15,18,19,21,24,26,27,29,30,31,32,34,35,36,37,38,42,43,44,45,46,50,55,
                          63,64,65,66,67,72,73,74,75,77,78,79,80,88,91,93,95,97,98,99]
        unseen_classes = random.sample(animal_classes, 10)

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_add(self.trainset, self.testset, self.unknownset, seen_classes,
                                                                      unseen_classes, self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset


class CIFARAdd50_Dataset(Dataset):
    def __init__(self):
        self.trainset = CIFAR10(root='./data', train=True, download=True)
        self.testset = CIFAR10(root='./data', train=False, download=True)
        self.unknownset = CIFAR100(root='./data', train=False, download=True)
        self.classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                     'truck': 9}

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        # 4 non animal
        seen_classes = [0, 1, 8, 9]
        # animal in CIFAR100
        animal_classes = [1,2,3,4,6,7,11,14,15,18,19,21,24,26,27,29,30,31,32,34,35,36,37,38,42,43,44,45,46,50,55,
                          63,64,65,66,67,72,73,74,75,77,78,79,80,88,91,93,95,97,98,99]
        unseen_classes = random.sample(animal_classes, 50)

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_add(self.trainset, self.testset, self.unknownset, seen_classes,
                                                                      unseen_classes, self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset

class CIFARAddN_Dataset(Dataset):
    def __init__(self):
        self.trainset = CIFAR10(root='./data', train=True, download=True)
        self.testset = CIFAR10(root='./data', train=False, download=True)
        self.unknownset = CIFAR100(root='./data', train=False, download=True)
        self.classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                     'truck': 9}

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        # 4 non animal
        seen_classes = [0, 1, 8, 9]
        # animal in CIFAR100
        animal_classes = [1,2,3,4,6,7,11,14,15,18,19,21,24,26,27,29,30,31,32,34,35,36,37,38,42,43,44,45,46,50,55,
                          63,64,65,66,67,72,73,74,75,77,78,79,80,88,91,93,95,97,98,99]
        unseen_classes = random.sample(animal_classes, args.unseen_num)

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_add(self.trainset, self.testset, self.unknownset, seen_classes,
                                                                      unseen_classes, self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset


class SVHN_Dataset(Dataset):
    def __init__(self):
        self.trainset = SVHN(root='./data', split='train', download=True)
        self.testset = SVHN(root='./data', split='test', download=True)
        self.classDict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                     '9': 9}
        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0),
            transforms.ToTensor(),
            transforms.Normalize([0.4377, 0.4438, 0.4728],
                                 [0.1980, 0.2010, 0.1970]),
        ])
        self.transform_test = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.4377, 0.4438, 0.4728],
                                                    [0.1980, 0.2010, 0.1970])])

    def sampler(self, seed, args):
        if seed is not None:
            random.seed(seed)
        seen_classes = random.sample(range(0, 10), 6)
        unseen_classes = [idx for idx in range(10) if idx not in seen_classes]

        osr_trainset, osr_valset, osr_testset = construct_ocr_dataset_aug(self.trainset, self.testset,
                                                                      seen_classes, unseen_classes,
                                                                      self.transform_train, self.transform_test, args)

        return osr_trainset, osr_valset, osr_testset



def construct_ocr_dataset(trainset, testset, seen_classes, unseen_classes, transform, args):
    if args.dataset in ['MNIST', 'CIFAR10']:
        osr_trainset = DatasetBuilder(
            [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
            transform)

        osr_valset = DatasetBuilder(
            [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
            transform)

        osr_testset = DatasetBuilder(
            [get_class_i(testset.data, testset.targets, idx) for idx in unseen_classes],
            transform)

    elif args.dataset in ['SVHN']:
        osr_trainset = DatasetBuilder(
            [get_class_i(trainset.data, trainset.labels, idx) for idx in seen_classes],
            transform)

        osr_valset = DatasetBuilder(
            [get_class_i(testset.data, testset.labels, idx) for idx in seen_classes],
            transform)

        osr_testset = DatasetBuilder(
            [get_class_i(testset.data, testset.labels, idx) for idx in unseen_classes],
            transform)

    return osr_trainset, osr_valset, osr_testset


def construct_ocr_dataset_aug(trainset, testset, seen_classes, unseen_classes, transform_train, transform_test, args):
    if args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100']:
        osr_trainset = DatasetBuilder(
            [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
            transform_train)

        osr_valset = DatasetBuilder(
            [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
            transform_test)

        osr_testset = DatasetBuilder(
            [get_class_i(testset.data, testset.targets, idx) for idx in unseen_classes],
            transform_test)

    elif args.dataset in ['SVHN']:
        osr_trainset = DatasetBuilder(
            [get_class_i(trainset.data, trainset.labels, idx) for idx in seen_classes],
            transform_train)

        osr_valset = DatasetBuilder(
            [get_class_i(testset.data, testset.labels, idx) for idx in seen_classes],
            transform_test)

        osr_testset = DatasetBuilder(
            [get_class_i(testset.data, testset.labels, idx) for idx in unseen_classes],
            transform_test)


    return osr_trainset, osr_valset, osr_testset


def construct_ocr_dataset_add(trainset, testset, unknownset, seen_classes, unseen_classes, transform_train, transform_test, args):

    osr_trainset = DatasetBuilder(
        [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
        transform_train)

    osr_valset = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
        transform_test)

    osr_testset = DatasetBuilder(
        [get_class_i(unknownset.data, unknownset.targets, idx) for idx in unseen_classes],
        transform_test)

    return osr_trainset, osr_valset, osr_testset


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    return x_i

# borrow from https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
class DatasetBuilder(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)

        img = self.datasets[class_label][index_wrt_class]

        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy(), mode='L')
        elif type(img).__module__ == np.__name__:
            if np.argmin(img.shape) == 0:
                img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
        elif isinstance(img, tuple): #ImageNet
            # img = Image.open(img[0])
            img = cv2.imread(img[0])
            img = Image.fromarray(img)

        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class
