import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy

from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch, BatchPAD


def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    try:
        return getattr(importlib.import_module('libcity.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        try:
            return getattr(importlib.import_module('libcity.data.dataset.dataset_subclass'),
                           config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False):
    """create dataloader(train/test/eval)

    Args:
        train_data (list of input): training data. Each element in data is a single input of the model. input is a list, which stores single input and target.
        eval_data (list of input): Verify data. Each element in data is a single input of the model. Input is a list, which stores single input and target.
        test_data (list of input): test data, each element in data is a single input of the model, input is a list, which stores single input and target
        feature_name(dict): Describe the feature name corresponding to each element of the input above. It should be ensured that len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): If the last batch does not meet the batch_size, whether to pad (use the last element to repeatedly pad).

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)"""
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True):
    """create dataloader(train/test/eval)

    Args:
        train_data (list of input): training data. Each element in data is a single input of the model. input is a list, which stores single input and target.
        eval_data (list of input): Verify data. Each element in data is a single input of the model. Input is a list, which stores single input and target.
        test_data (list of input): test data, each element in data is a single input of the model, input is a list, which stores single input and target
        feature_name(dict): Describe the feature name corresponding to each element of the input above. It should be ensured that len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): Used to pad features of variable length to the same length. Each feature name is used as a key. If a feature name is not in the dict, padding will not be performed.
        pad_max_len(dict): used to intercept features of variable length and cut features that are too long
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)"""
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = BatchPAD(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    return train_dataloader, eval_dataloader, test_dataloader
