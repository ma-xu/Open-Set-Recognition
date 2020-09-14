from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, train_class_num=5, test_class_num=7, includes_all_train_class=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self._update_open_set(train_class_num,test_class_num, includes_all_train_class)

    def _update_open_set(self, train_class_num=5, test_class_num=5, includes_all_train_class=False):
        """
        Author: Xu Ma (xuma@my.unt.edu)
        Date: Sep/13/2020
        :param train_class_num: known class number in training
        :param test_class_num: original class number involoved in testing
        :param includes_all_train_class: If contains all unknown classes in testing.
        :return: Update the data, targets,classes and class_to_idx; add openness attribute.
        """
        assert train_class_num > 0 and train_class_num < len(self.classes)
        if includes_all_train_class:
            assert test_class_num > train_class_num  # not include equal to ensure openness.
        rnd = np.random.RandomState(42)  # Ensure identical results.
        class_list = list(range(len(self.classes)))
        train_classes = rnd.choice(class_list, train_class_num, replace=False).tolist()
        if includes_all_train_class:
            rnd = np.random.RandomState(42)
            test_classes = rnd.choice(class_list, test_class_num, replace=False).tolist()
        else:
            test_classes = rnd.choice(class_list, test_class_num, replace=False).tolist()

        # rest_class_list = [i for i in class_list if i not in train_classes]
        # if includes_all_train_class:
        #     test_classes = rnd.choice(rest_class_list, test_class_num-train_class_num, replace=False).tolist()
        #     test_classes = train_classes+test_classes
        # else:
        #     test_classes = rnd.choice(rest_class_list, test_class_num, replace=False).tolist()

        # Update self.classes
        selected_elements = [self.classes[index] for index in train_classes]
        selected_elements.append('unknown')
        self.classes = selected_elements
        # update self.class_to_idx
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        # Processing data
        if self.train:
            indexes = [i for i, x in enumerate(self.targets) if x in train_classes]
            self.data = self.data[indexes, :, :, :]
            self.targets =[train_classes.index(self.targets[i]) for i in indexes]
            print(f"Training data includes {train_class_num} classes, {len(self.targets)} samples.")
        else:
            indexes = [i for i, x in enumerate(self.targets) if x in test_classes]
            self.data = self.data[indexes, :, :, :]
            temp_test_classes = [x for x in test_classes if x not in train_classes]
            train_and_test_calsses = train_classes+temp_test_classes
            self.targets = [train_and_test_calsses.index(self.targets[i]) for i in indexes]
            for i in range(0,len(self.targets)) :
                if self.targets[i]>train_class_num:
                    self.targets[i] = train_class_num
            print(f"Testing data includes {train_class_num+1} classes, {len(self.targets)} samples.")
            self.openness= float(len(temp_test_classes))/float(train_class_num+len(temp_test_classes))
            print(f"During testing, openness is {self.openness}.")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


if __name__ == '__main__':
    trainset = CIFAR100(root='./data', train=False, download=True)
