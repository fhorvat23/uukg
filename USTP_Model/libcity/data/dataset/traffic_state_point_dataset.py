import os

from libcity.data.dataset import TrafficStateDataset


class TrafficStatePointDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        """Load .geo file, format [geo_id, type, coordinates, properties (several columns)]"""
        super()._load_geo()

    def _load_rel(self):
        """Load .rel file, format [rel_id, type, origin_id, destination_id, properties (several columns)]

        Returns:
            np.ndarray: self.adj_mx, N*N adjacency matrix"""
        super()._load_rel()

    def _load_dyna(self, filename):
        """Load .dyna file, format [dyna_id, type, time, entity_id, properties (several columns)]
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 3d-array (len_time, num_nodes, feature_dim)"""
        return super()._load_dyna_3d(filename)

    def _add_external_information(self, df, ext_data=None):
        """Add external information (day of week/day of week, time of day/time of day, external data)

        Args:
            df(np.ndarray): multidimensional array of traffic status data, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): external data

        Returns:
            np.ndarray: fused external data and traffic status data, (len_time, num_nodes, feature_dim_plus)"""
        return super()._add_external_information_3d(df, ext_data)

    def get_data_feature(self):
        """Returns the characteristics of the data set, scaler is the normalization method, adj_mx is the adjacency matrix, num_nodes is the number of points,
        feature_dim is the dimension of the input data, and output_dim is the dimension of the model output.

        Returns:
            dict: A dictionary containing relevant features of the dataset"""
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}
