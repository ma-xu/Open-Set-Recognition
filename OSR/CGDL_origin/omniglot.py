import numpy as np
import os
import cv2
import json
import torch
import torchvision

class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError

class OmniglotLoader(BaseDataLoader):
    def __init__(self, batch_size=64, train=False, shuffle=True, drop_last=False):
        super(OmniglotLoader, self).__init__(batch_size, train, shuffle, drop_last)
        omniglot_path = 'data/omn_dataset'

        if os.path.isdir(omniglot_path):
            print('Files already downloaded and verified')
        else:
            raise FileNotFoundError('Omniglot dataset not found. Please download it and put it under \'{}\''.format(omniglot_path))

        images = []
        labels = []
        self._len = 0
        self.task_dataloader = []
        self.num_classes = []

        for p in [os.path.join(omniglot_path, 'images_background'), os.path.join(omniglot_path, 'images_evaluation')]:
            for task_path in sorted(os.listdir(p)):
                task_path = os.path.join(p, task_path)
                task_images = []
                task_labels = []
                for i, cls_path in enumerate(sorted(os.listdir(task_path))):
                    cls_path = os.path.join(task_path, cls_path)
                    ims = [cv2.resize(cv2.imread(os.path.join(cls_path, filename), cv2.IMREAD_GRAYSCALE), (28,28)) / 255 for filename in sorted(os.listdir(cls_path))]

                    if train:
                        ims = ims[:int(len(ims)*0.69)]
                    else:
                        ims = ims[int(len(ims)*0.69):]

                    self._len += len(ims)
                    task_images += ims
                    task_labels += [i for _ in range(len(ims))]

                task_images = np.expand_dims(task_images, 1)
                dataset = CustomDataset(data=torch.Tensor(task_images).float(), labels=torch.Tensor(task_labels).long())
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle,
                                                         drop_last=drop_last)
                self.task_dataloader.append(dataloader)
                self.num_classes.append(len(np.unique(task_labels)))

                images.append(task_images)
                labels.append(task_labels)

        images = np.concatenate(images)
        labels = np.concatenate(labels)

        new_label = 0
        new_labels = [new_label]
        for prev_label, label in zip(labels[:-1], labels[1:]):
            if prev_label != label:
                new_label += 1
            new_labels.append(new_label)

        new_labels = torch.Tensor(new_labels).long()
        images = torch.from_numpy(images).float()

        dataset = CustomDataset(data=images, labels=new_labels)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)

        self.labels = []
        cnter = 0
        for num_classes in self.num_classes:
            self.labels.append(list(range(cnter, cnter + num_classes)))
            cnter += num_classes

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(50)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(50))
        else:
            assert task in list(range(50)), 'Unknown task: {}'.format(task)
            return self.labels[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 1


    @property
    def num_classes_single(self):
        return sum(self.num_classes)


    @property
    def num_classes_multi(self):
        return self.num_classes
