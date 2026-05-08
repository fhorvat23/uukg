import torch
import numpy as np


class Batch(object):

    def __init__(self, feature_name):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
        """
        self.data = {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        """append a new item into the batch

        Args:
            item (list): a group of inputs, in the same order as feature_name, feature_name is the name of this group of inputs"""
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])

    def to_tensor(self, device):
        """Transfer data self.data to device

        Args:
            device(torch.device): GPU/CPU device"""
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_ndarray, only support int, float but you give {}'.format(self.feature_name[key]))


class BatchPAD(Batch):

    def __init__(self, feature_name, pad_item=None, pad_max_len=None):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
            pad_item (dict): key is the feature name, and value is the padding
            value. We will just padding the feature in pad_item
            pad_max_len (dict): key is the feature name, and value is the max
                length of padded feature. use this parameter to truncate the
                feature.
        """
        super().__init__(feature_name=feature_name)
        # The default is to pad according to the longest length of each feature in the batch. If the length of a feature exceeds pad_max_len, it will be cut.
        self.pad_len = {}
        self.origin_len = {}  # Used to know the original length of the track before completion
        self.pad_max_len = pad_max_len if pad_max_len is not None else {}
        self.pad_item = pad_item if pad_item is not None else {}
        for key in feature_name:
            self.data[key] = []
            if key in self.pad_item:
                self.pad_len[key] = 0
                self.origin_len[key] = []

    def append(self, item):
        """append a new item into the batch

        Args:
            item (list): a group of inputs, in the same order as feature_name, feature_name is the name of this group of inputs"""
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            # It is necessary to ensure that the order of each feature of the item is consistent with the order of the features in feature_name passed in during initialization.
            self.data[key].append(item[i])
            if key in self.pad_item:
                self.origin_len[key].append(len(item[i]))
                if self.pad_len[key] < len(item[i]):
                    # Keep pad_len as large as possible
                    self.pad_len[key] = len(item[i])

    def padding(self):
        """Only provides features for one-dimensional arrays"""
        for key in self.pad_item:
            # Only fill the features in pad_item
            if key not in self.data:
                raise KeyError('when pad a batch, raise this error!')
            max_len = self.pad_len[key]
            if key in self.pad_max_len:
                max_len = min(self.pad_max_len[key], max_len)
            for i in range(len(self.data[key])):
                if len(self.data[key][i]) < max_len:
                    self.data[key][i] += [self.pad_item[key]] * \
                        (max_len - len(self.data[key][i]))
                else:
                    # The principle of interception is to discard the previous points
                    # Because it’s a time series
                    self.data[key][i] = self.data[key][i][-max_len:]
                    # For the cut, we can't restore it, but at least we don't make it wrong
                    self.origin_len[key][i] = max_len

    def get_origin_len(self, key):
        return self.origin_len[key]

    def to_tensor(self, device):
        """Transfer data self.data to device

        Args:
            device(torch.device): GPU/CPU device"""
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'array of int':
                for i in range(len(self.data[key])):
                    for j in range(len(self.data[key][i])):
                        try:
                            self.data[key][i][j] = torch.LongTensor(np.array(self.data[key][i][j])).to(device)
                        except TypeError:
                            print('device is ', device)
                            exit()
            elif self.feature_name[key] == 'no_pad_int':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.LongTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_pad_float':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.FloatTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_tensor':
                pass
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))
