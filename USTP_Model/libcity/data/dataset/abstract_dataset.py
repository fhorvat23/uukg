class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        """DataLoader that returns data, including training data, test data, and verification data

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)"""
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        """Returns a dict containing relevant features of the dataset

        Returns:
            dict: A dictionary containing relevant features of the dataset"""
        raise NotImplementedError("get_data_feature not implemented")
