class AbstractExecutor(object):

    def __init__(self, config, model, data_feature):
        raise NotImplementedError("Executor not implemented")

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor train not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")

    def load_model(self, cache_name):
        """Load the cache of the corresponding model

        Args:
            cache_name(str): saved file name"""
        raise NotImplementedError("Executor load cache not implemented")

    def save_model(self, cache_name):
        """Save current model to file

        Args:
            cache_name(str): saved file name"""
        raise NotImplementedError("Executor save cache not implemented")
