from libcity.model.abstract_model import AbstractModel


class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        super().__init__(config, data_feature)

    def predict(self, batch):
        """Input a batch of data and return the corresponding prediction value, which should generally be the result of **multi-step prediction**. Generally, the forward() method of nn.Moudle will be called.

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch"""

    def calculate_loss(self, batch):
        """Enter a batch of data and return the loss of the training process, which means you need to define a loss function.

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss"""
