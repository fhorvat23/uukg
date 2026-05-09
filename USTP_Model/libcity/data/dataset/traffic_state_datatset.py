import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


class TrafficStateDataset(AbstractDataset):
    """Base class for traffic state prediction dataset.
    By default, the data of `input_window` is used to predict the data corresponding to `output_window`, that is, an X and a y.
    Generally, external data is integrated into X for joint prediction, so the data is [X, y].
    By default, `train_rate` and `eval_rate` are used to directly divide the training set, test set, and validation set in the dimension of sample number (num_samples)."""

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.ext_scaler_type = self.config.get('ext_scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' \
            + str(self.config.get('kg_model', 'none'))
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'traffic_state_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        # Load the config.json file of the data set
        self.weight_col = self.config.get('weight_col', '')
        self.data_col = self.config.get('data_col', '')
        self.ext_col = self.config.get('ext_col', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.data_files = self.config.get('data_files', self.dataset)
        self.ext_file = self.config.get('ext_file', self.dataset)
        self.output_dim = self.config.get('output_dim', 1)
        self.time_intervals = self.config.get('time_intervals', 300)  # s
        self.init_weight_inf_or_zero = self.config.get('init_weight_inf_or_zero', 'inf')
        self.set_weight_link_or_dist = self.config.get('set_weight_link_or_dist', 'dist')
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        self.calculate_weight_adj = self.config.get('calculate_weight_adj', False)
        self.weight_adj_epsilon = self.config.get('weight_adj_epsilon', 0.1)
        self.distance_inverse = self.config.get('distance_inverse', False)
        self.kg_model = self.config.get('kg_model', None)
        self.embed_dir = self.config.get(
            'embed_dir', '../UrbanKG_Embedding_Model/UrbanKG_Embedding')

        # initialization
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}  # The inputs of this class are only X and y
        self.adj_mx = None
        self.scaler = None
        self.ext_scaler = None
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_nodes = 0
        self.num_batches = 0
        self.node_embeddings = None
        self._logger = getLogger()
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):  # .rel file is not necessary
            self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        # Load KG node embeddings after num_nodes is known from geo file
        self.node_embeddings = self._load_kg_node_embeddings(self.num_nodes)

    def _load_geo(self):
        """Load .geo file, format [geo_id, type, coordinates, properties (several columns)]"""
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))

    def _load_grid_geo(self):
        """Load .geo file, format [geo_id, type, coordinates, row_id, column_id, properties (several columns)]"""
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.geo_to_rc = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        for i in range(geofile.shape[0]):
            self.geo_to_rc[geofile['geo_id'][i]] = [geofile['row_id'][i], geofile['column_id'][i]]
        self.len_row = max(list(geofile['row_id'])) + 1
        self.len_column = max(list(geofile['column_id'])) + 1
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_grids=' + str(len(self.geo_ids))
                          + ', grid_size=' + str((self.len_row, self.len_column)))

    def _load_rel(self):
        """Load .rel file, format [rel_id, type, origin_id, destination_id, properties (several columns)],
        Generate an N*N adjacency matrix. The calculation logic is as follows:
        (1) The column name corresponding to the weight is specified with the global parameter `weight_col`, \
        (2) If this parameter is not specified, \
            (2.1) rel has only 4 columns, then each row in rel is considered to represent an adjacent edge with a weight of 1. The remaining edge weights are 0, which means they are not adjacent. \
            (2.2) rel has only 5 columns, then the default last column is `weight_col` \
            (2.3) Otherwise, an error will be reported \
        (3) Calculate the adjacency matrix based on the obtained weight column `weight_col` \
            (3.1) Parameter `bidir_adj_mx`=True means constructing an undirected graph, =False means directed graph\
            (3.2) If the parameter `set_weight_link_or_dist` is `link`, it means constructing a 01 matrix, and if it is `dist`, it means constructing a weight matrix (non-01) \
            (3.3) If the parameter `init_weight_inf_or_zero` is `zero`, it means that the matrix is initialized to all 0s, and `inf` means that the matrix is initialized to all inf. The initialization value is the weight of the edge that does not exist in the rel file\
            (3.4) The parameter `calculate_weight_adj`=True indicates that the Gaussian kernel function with a threshold is applied to the weight matrix to sparse it, and the 01 matrix is not processed. =False does not sparse it.
            Modify the function self._calculate_adjacency_matrix() to construct other methods to replace the sparseness method of the full threshold Gaussian kernel\

        Returns:
            np.ndarray: self.adj_mx, N*N adjacency matrix"""
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':  # Confirm weight column based on weight_col
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties has more than one column, and weight_col is not specified, an error is reported.
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # The 4 columns indicate that there is no properties column, that is, there are some in the rel file that represent adjacent ones, otherwise they are not adjacent.
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = 'link'
                self.init_weight_inf_or_zero = 'zero'
                self.distance_df = relfile[['origin_id', 'destination_id']]
            else:  # len(relfile.columns) == 5, properties has only one column, then by default this column is the weight column
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # Convert data into matrix form
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # Keep the original distance value
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' only retains the adjacency of 01
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # Calculate weight
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != 'link':
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != 'link':
            self._calculate_adjacency_matrix()

    def _load_grid_rel(self):
        """Construct an adjacency matrix based on the grid structure. One grid is adjacent to 8 grids around it.

        Returns:
            np.ndarray: self.adj_mx, N*N adjacency matrix"""
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(self.len_row):
            for j in range(self.len_column):
                index = i * self.len_column + j  # grid_id
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < self.len_row and nei_j >= 0 and nei_j < self.len_column:
                        nei_index = nei_i * self.len_column + nei_j  # neighbor_grid_id
                        self.adj_mx[index][nei_index] = 1
                        self.adj_mx[nei_index][index] = 1
        self._logger.info("Generate grid rel file, shape=" + str(self.adj_mx.shape))

    def _calculate_adjacency_matrix(self):
        """Use a Gaussian kernel with a threshold to calculate the weight of the adjacency matrix. If there are other calculation methods, you can override this function,
        The formula is: $ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ is the variance,
        Values less than the threshold `weight_adj_epsilon` are set to 0: $ w_{ij}[w_{ij}<\epsilon]=0 $

        Returns:
            np.ndarray: self.adj_mx, N*N adjacency matrix"""
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def _distance_inverse(self):
        self._logger.info("Start Calculate the weight by _distance_inverse!")
        self.adj_mx = 1 / self.adj_mx
        self.adj_mx[np.isinf(self.adj_mx)] = 1

    def _load_dyna(self, filename):
        """Load data files (.dyna/.grid/.od/.gridod). Subclasses must implement this method to specify how to load data files and return corresponding multidimensional data.
        Provides 5 well-implemented methods to load the above types of files and convert them into arrays of different shapes:
        `_load_dyna_3d`/`_load_grid_3d`/`_load_grid_4d`/`_load_grid_od_4d`/`_load_grid_od_6d`

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array"""
        raise NotImplementedError('Please implement the function `_load_dyna()`.')

    def _load_dyna_3d(self, filename):
        """Load .dyna file, format [dyna_id, type, time, entity_id, properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 3d-array: (len_time, num_nodes, feature_dim)"""
        # Load dataset
        self._logger.info("Loading file " + filename + '.dyna')
        dynafile = pd.read_csv(self.data_path + filename + '.dyna')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dynafile = dynafile[data_col]
        else:  # If not specified, all columns are loaded.
            dynafile = dynafile[dynafile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # Convert 3-D array
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
        return data

    def _load_grid_3d(self, filename):
        """Load .grid file, format [dyna_id, type, time, row_id, column_id, properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 3d-array: (len_time, num_grids, feature_dim)"""
        # Load dataset
        self._logger.info("Loading file " + filename + '.grid')
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:  # If not specified, all columns are loaded.
            gridfile = gridfile[gridfile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # Convert 3-D array
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_grid_4d(self, filename):
        """Load .grid file, format [dyna_id, type, time, row_id, column_id, properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 4d-array: (len_time, len_row, len_column, feature_dim)"""
        # Load dataset
        self._logger.info("Loading file " + filename + '.grid')
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:  # If not specified, all columns are loaded.
            gridfile = gridfile[gridfile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # Convert 4-D array
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(self.len_row):
            tmp = []
            for j in range(self.len_column):
                index = (i * self.len_column + j) * len_time
                tmp.append(df[index:index + len_time].values)
            data.append(tmp)
        data = np.array(data, dtype=np.float)  # (len_row, len_column, len_time, feature_dim)
        data = data.swapaxes(2, 0).swapaxes(1, 2)  # (len_time, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_od_4d(self, filename):
        """Load .od file, format [dyna_id, type, time, origin_id, destination_id properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 4d-array: (len_time, len_row, len_column, feature_dim)"""
        self._logger.info("Loading file " + filename + '.od')
        odfile = pd.read_csv(self.data_path + filename + '.od')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_id')
            data_col.insert(2, 'destination_id')
            odfile = odfile[data_col]
        else:  # If not specified, all columns are loaded.
            odfile = odfile[odfile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(odfile['time'][:int(odfile.shape[0] / self.num_nodes / self.num_nodes)])
        self.idx_of_timesolts = dict()
        if not odfile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx

        feature_dim = len(odfile.columns) - 3
        df = odfile[odfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.num_nodes, self.num_nodes, len_time, feature_dim))
        for i in range(self.num_nodes):
            origin_index = i * len_time * self.num_nodes  # Each starting point occupies len_t*n lines
            for j in range(self.num_nodes):
                destination_index = j * len_time  # Each endpoint occupies len_t rows
                index = origin_index + destination_index
                data[i][j] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_nodes, num_nodes, feature_dim)
        self._logger.info("Loaded file " + filename + '.od' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_4d(self, filename):
        """Load .gridod file, format [dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 4d-array: (len_time, num_grids, num_grids, feature_dim)"""
        # Load dataset
        self._logger.info("Loading file " + filename + '.gridod')
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:  # If not specified, all columns are loaded.
            gridodfile = gridodfile[gridodfile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # Convert 4-D array
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((len(self.geo_ids), len(self.geo_ids), len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # Each starting point occupies len_t*n lines
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # Each endpoint occupies len_t rows
                        index = origin_index + destination_index
                        # print(index, index + len_time)
                        # print((oi, oj), (di, dj))
                        # print(oi * self.len_column + oj, di * self.len_column + dj)
                        data[oi * self.len_column + oj][di * self.len_column + dj] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_grids, num_grids, feature_dim)
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_6d(self, filename):
        """Load .gridod file, format [dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties (several columns)],
        The order of IDs in the .geo file should be consistent with that in .dyna.
        The global parameter `data_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Args:
            filename(str): data file name, excluding suffix

        Returns:
            np.ndarray: data array, 6d-array: (len_time, len_row, len_column, len_row, len_column, feature_dim)"""
        # Load dataset
        self._logger.info("Loading file " + filename + '.gridod')
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':  # Load a dataset based on specified columns
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:  # If not specified, all columns are loaded.
            gridodfile = gridodfile[gridodfile.columns[2:]]  # All columns starting from time column
        # Find time series
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():  # Time has no null value
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # Convert 6-d array
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.len_row, self.len_column, self.len_row, self.len_column, len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # Each starting point occupies len_t*n lines
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # Each endpoint occupies len_t rows
                        index = origin_index + destination_index
                        # print(index, index + len_time)
                        data[oi][oj][di][dj] = df[index:index + len_time].values
        data = data.transpose((4, 0, 1, 2, 3, 5))  # (len_time, len_row, len_column, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_ext(self):
        """Load .ext file, format [ext_id, time, properties (several columns)],
        The global parameter `ext_col` is used to specify the columns of data that need to be loaded. If not set, all will be loaded by default.

        Returns:
            np.ndarray: external data array, shape: (timeslots, ext_dim)"""
        # Load dataset
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        if self.ext_col != '':  # Load a dataset based on specified columns
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:  # str
                ext_col = [self.ext_col].copy()
            ext_col.insert(0, 'time')
            extfile = extfile[ext_col]
        else:  # If not specified, all columns are loaded.
            extfile = extfile[extfile.columns[1:]]  # All columns starting from time column
        # Find time series
        self.ext_timesolts = extfile['time']
        self.idx_of_ext_timesolts = dict()
        if not extfile['time'].isna().any():  # Time has no null value
            self.ext_timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.ext_timesolts))
            self.ext_timesolts = np.array(self.ext_timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.ext_timesolts):
                self.idx_of_ext_timesolts[_ts] = idx
        # Find external feature array
        feature_dim = len(extfile.columns) - 1
        df = extfile[extfile.columns[-feature_dim:]].values
        self._logger.info("Loaded file " + self.ext_file + '.ext' + ', shape=' + str(df.shape))
        return df

    def _add_external_information(self, df, ext_data=None):
        """Combine external data and original traffic status data into a high-dimensional array. Subclasses must implement this method to specify how to merge external data and traffic status data.
        If you don’t want to add external data, you can return the traffic status data `df` directly.
        Provides 3 well-implemented methods suitable for combining traffic status data of different shapes with external data:
        `_add_external_information_3d`/`_add_external_information_4d`/`_add_external_information_6d`

        Args:
            df(np.ndarray): multidimensional array of traffic status data
            ext_data(np.ndarray): external data

        Returns:
            np.ndarray: fused external data and traffic status data"""
        raise NotImplementedError('Please implement the function `_add_external_information()`.')

    def _load_kg_node_embeddings(self, num_nodes):
        """Load region, POI, and road KG embeddings and aggregate them to the node (region) level.

        The city prefix is derived from the first 3 characters of ``self.dataset``
        (e.g. 'NYC' from 'NYCTaxi20200406').  The KG model name comes from
        ``self.kg_model`` (e.g. 'RefH'), producing filenames like
        ``NYC_RefH_region_embeddings.npy``.

        POI and road embeddings are stored per-entity with the region_id appended as
        the last column.  They are mean-pooled to region level before concatenation.

        Args:
            num_nodes (int): number of spatial nodes in the dataset

        Returns:
            np.ndarray: shape (num_nodes, 3 * embed_dim) concatenating
                [region_emb | poi_emb | road_emb], or None if kg_model is not set."""
        if self.kg_model is None:
            return None

        city = self.dataset[:3]  # 'NYC' or 'CHI'
        city_dir = os.path.join(self.embed_dir, city, self.kg_model)
        prefix = '{}_{}_'.format(city, self.kg_model)

        region_path = os.path.join(city_dir, '{}region_embeddings.npy'.format(prefix))
        poi_path    = os.path.join(city_dir, '{}poi_embeddings.npy'.format(prefix))
        road_path   = os.path.join(city_dir, '{}road_embeddings.npy'.format(prefix))

        for p in (region_path, poi_path, road_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    'KG embedding file not found: {}. '
                    'Run get_embedding.py with --model {} --dataset {} first.'
                    .format(p, self.kg_model, city))

        # Region embeddings: (num_regions, embed_dim) — infer dim from file
        region_raw = np.load(region_path).astype(np.float32)
        embed_dim = region_raw.shape[1]
        region_emb = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        n = min(region_raw.shape[0], num_nodes)
        region_emb[:n] = region_raw[:n]

        # POI embeddings: (num_pois, embed_dim + 1), last col = region_id
        poi_raw = np.load(poi_path).astype(np.float32)
        poi_emb = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        poi_counts = np.zeros(num_nodes, dtype=np.int32)
        for row in poi_raw:
            rid = int(row[embed_dim])
            if 0 <= rid < num_nodes:
                poi_emb[rid] += row[:embed_dim]
                poi_counts[rid] += 1
        mask = poi_counts > 0
        poi_emb[mask] /= poi_counts[mask, None]

        # Road embeddings: (num_roads, embed_dim + 1), last col = region_id
        road_raw = np.load(road_path).astype(np.float32)
        road_emb = np.zeros((num_nodes, embed_dim), dtype=np.float32)
        road_counts = np.zeros(num_nodes, dtype=np.int32)
        for row in road_raw:
            rid = int(row[embed_dim])
            if 0 <= rid < num_nodes:
                road_emb[rid] += row[:embed_dim]
                road_counts[rid] += 1
        mask = road_counts > 0
        road_emb[mask] /= road_counts[mask, None]

        self._logger.info(
            'Loaded KG embeddings: model={}, city={}, embed_dim={}, '
            'total_node_dim={}'.format(self.kg_model, city, embed_dim, 3 * embed_dim))

        return np.concatenate([region_emb, poi_emb, road_emb], axis=1)  # (num_nodes, 3*embed_dim)

    def _add_external_information_3d(self, df, ext_data=None):
        """Add external information (day of week/day of week, time of day/time of day, external data)

        Args:
            df(np.ndarray): multidimensional array of traffic status data, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): external data

        Returns:
            np.ndarray: fused external data and traffic status data, (len_time, num_nodes, feature_dim_plus)"""
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)
        # external dataset
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim selects the data of the required time step
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:  # No specific timestamp is given. Only external data and original data of equal length can be connected together by default.
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)

        # node_emb = self._load_kg_node_embeddings(num_nodes)
        # if node_emb is not None:
        #     new_data = np.empty(
        #         (data.shape[0], num_nodes, data.shape[2] + node_emb.shape[1]),
        #         dtype=np.float32)
        #     for t in range(data.shape[0]):
        #         new_data[t] = np.concatenate([data[t], node_emb], axis=1)
        #     return new_data

        return data

    def _add_external_information_4d(self, df, ext_data=None):
        """Add external information (day of week/day of week, time of day/time of day, external data)

        Args:
            df(np.ndarray): Multidimensional array of traffic status data, (len_time, len_row, len_column, feature_dim)
            ext_data(np.ndarray): external data

        Returns:
            np.ndarray: fused external data and traffic status data, (len_time, len_row, len_column, feature_dim_plus)"""
        num_samples, len_row, len_column, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, dayofweek] = 1
            data_list.append(day_in_week)
        # external dataset
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim selects the data of the required time step
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                    data_list.append(data_ind)
            else:  # No specific timestamp is given. Only external data and original data of equal length can be connected together by default.
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _add_external_information_6d(self, df, ext_data=None):
        """Add external information (day of week/day of week, time of day/time of day, external data)

        Args:
            df(np.ndarray): Multidimensional array of traffic status data,
                (len_time, len_row, len_column, len_row, len_column, feature_dim)
            ext_data(np.ndarray): external data

        Returns:
            np.ndarray: fused external data and traffic status data,
            (len_time, len_row, len_column, len_row, len_column, feature_dim)"""
        num_samples, len_row, len_column, _, _, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                transpose((5, 1, 2, 3, 4, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, :, :, dayofweek] = 1
            data_list.append(day_in_week)
        # external dataset
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim selects the data of the required time step
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                        transpose((5, 1, 2, 3, 4, 0))
                    data_list.append(data_ind)
            else:  # No specific timestamp is given. Only external data and original data of equal length can be connected together by default.
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                            transpose((5, 1, 2, 3, 4, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _generate_input_data(self, df):
        """Split the input according to the global parameters `input_window` and `output_window` to generate the tensor input required by the model,
        That is, using the past `input_window` length time series to predict the future `output_window` length time series

        Args:
            df(np.ndarray): data array, shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): model input data, (epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): model output data, (epoch_size, output_length, ..., feature_dim)"""
        num_samples = df.shape[0]
        # The length of the past time window used for prediction depends on self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        # The length of the future time window depends on self.output_window
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        if len(x) == 0:
            required_steps = self.input_window + self.output_window
            raise ValueError(
                "No training samples can be generated from dataset '{}' "
                "(time steps: {}, input_window: {}, output_window: {}, required minimum: {}). "
                "Use smaller window sizes or provide more time steps in the dyna/grid file."
                .format(self.dataset, num_samples, self.input_window, self.output_window, required_steps)
            )
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_data(self):
        """Load the data file (.dyna/.grid/.od/.gridod) and external data (.ext), fuse the two, and return them in the form of X, y

        Returns:
            tuple: tuple contains:
                x(np.ndarray): model input data, (num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): model output data, (num_samples, output_length, ..., feature_dim)"""
        # Dealing with multiple data file issues
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # Load external data
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # external dataset
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ..., output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _split_train_val_test(self, x, y):
        """Divide the training set, test set, validation set, and cache the data set

        Args:
            x(np.ndarray): input data (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): output data (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)"""
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        """Load the data set, divide it into training set, test set, validation set, and cache the data set

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)"""
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        """Load the previously cached training set, test set, and validation set

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)"""
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, scaler_type, x_train, y_train):
        """Select the data normalization method based on the global parameter `scaler_type`

        Args:
            x_train: training data X
            y_train: training data y

        Returns:
            Scaler: normalization object"""
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """DataLoader that returns data, including training data, test data, and verification data

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)"""
        # Load dataset
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        # Data normalization
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        # Aggregate the X and y of the training set into a list, and the same is true for the test set and verification set.
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i] is a tuple consisting of x_train[i] and y_train[i]
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        # Transfer Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """Returns the characteristics of the data set. Subclasses must implement this function and return the necessary characteristics.

        Returns:
            dict: A dictionary containing relevant features of the dataset"""
        raise NotImplementedError('Please implement the function `get_data_feature()`.')
