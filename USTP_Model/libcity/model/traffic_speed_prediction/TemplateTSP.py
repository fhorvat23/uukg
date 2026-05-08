from logging import getLogger
import torch
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class TemplateTSP(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """Construct model
        :param config: Configuration dictionary derived from various configurations
        :param data_feature: The necessary data-related features returned from the `get_data_feature()` interface of the Dataset class"""
        # 1. Initialize the parent class (required)
        super().__init__(config, data_feature)
        # 2. Obtain the desired information from data_feature. Note that different models use different Dataset classes, and the returned data_feature content is different (required)
        # Take TrafficStateGridDataset as an example to demonstrate data retrieval. The following data can be retrieved. If you do not need it, you do not need to retrieve it.
        # **These parameters cannot be taken from config**
        self._scaler = self.data_feature.get('scaler')  # for data normalization
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # adjacency matrix
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # Number of grids
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # input dimensions
        self.output_dim = self.data_feature.get('output_dim', 1)  # Output dimensions
        self.len_row = self.data_feature.get('len_row', 1)  # Number of grid rows
        self.len_column = self.data_feature.get('len_column', 1)  # Number of grid columns
        # 3. Initialize log for necessary output (required)
        self._logger = getLogger()
        # 4. Initialize device (required)
        self.device = config.get('device', torch.device('cpu'))
        # 5. Initialize the length of the input and output time steps (optional)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        # 6. Other parameters taken from config are mainly parameters used to construct the model structure (required)
        # These parameters related to the model structure should be placed in libcity/config/model/model_name.json (required)
        # For example: self.blocks = config['blocks']
        # ...
        # 7. Construct the hierarchical structure of the deep model (required)
        # For example: using a simple RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, batch):
        """Call the model to calculate the output corresponding to this batch input, the interface that nn.Module must implement
        :param batch: Input data, dictionary-like, data can be obtained by dictionary method
        :return:"""
        # 1. Get the data, assuming there are 4 types of data in the dictionary, X, y, X_ext, y_ext
        # Of course, you generally only need to take the input data, such as X, X_ext, because this function is used to calculate the output.
        # The feature dimension of the model input data should be equal to self.feature_dim
        # x = batch['X']  # shape = (batch_size, input_length, ..., feature_dim)
        # For example: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2. Calculate the output results of the model based on the input data
        # The feature dimension of the model output result should be equal to self.output_dim
        # Other dimensions of the model output results should be consistent with batch['y'], only the feature dimensions may be different (because batch['y'] may contain some external features)
        # If the model's single-step prediction, batch['y'] is multi-step data, the time dimension may also be different.
        # For example: outputs = self.model(x)
        # 3. Return the output result
        # For example: return outputs

    def calculate_loss(self, batch):
        """Input a batch of data and return the loss of the batch data during the training process, which means you need to define a loss function.
        :param batch: Input data, dictionary-like, data can be obtained by dictionary method
        :return: training loss (tensor)"""
        # 1. Get the true value ground_truth
        y_true = batch['y']
        # 2. Get the predicted value
        y_predicted = self.predict(batch)
        # 3. Use self._scaler to reversely normalize the normalized true values ​​and predicted values ​​(required)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4. Call the loss function to calculate the error between the true value and the predicted value.
        # Common loss functions are defined in libcity/model/loss.py
        # If the model source code uses the loss, it can be called directly, taking MSE as an example:
        res = loss.masked_mse_torch(y_predicted, y_true)
        # If the loss function used in the model source code is not in loss.py, you need to implement the loss function yourself.
        # ...(custom loss function)
        # 5. Return the result of loss
        return res

    def predict(self, batch):
        """Enter a batch of data and return the corresponding prediction value, which should generally be the result of **multi-step prediction**
        Generally, the forward() method defined above will be called.
        :param batch: Input data, dictionary-like, data can be obtained by dictionary method
        :return: predict result of this batch (tensor)"""
        # If the result of self.forward() meets the requirements, it can be returned directly
        # If it does not meet the requirements, for example, self.forward() performs a single time step prediction, but the model training uses multi-step predictions based on the data of each batch.
        # You can refer to the predict() function in libcity/model/traffic_speed_prediction/STGCN.py for multi-step prediction.
        # The principle of multi-step prediction is: make one-step prediction first, and use the result of the first-step prediction to make the second-step prediction, **instead of using the true value of the one-step prediction to make the second-step prediction!**
        # Take the result of self.forward() as an example:
        return self.forward(batch)
