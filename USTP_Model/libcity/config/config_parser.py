import os
import json
import torch


class ConfigParser(object):
    """use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    It is worth noting that the parameters of each stage are currently placed in the same dict, so it is necessary to ensure that the namespace does not conflict during programming.
    config priority: command line > config file > default config"""

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        """Args:
            task, model, dataset (str): three parameters that the user must specify on the command line
            config_file (str): The file name of the configuration file, which will be searched in the project root directory
            other_args (dict): other parameters passed in through the command line"""
        self.config = {}
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict)
        self._parse_config_file(config_file)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # Currently, it is tentatively determined that these three parameters must be specified by the user.
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = False if task == 'map_matching' else train
        if other_args is not None:
            # TODO: You can design and add parameter checking here, which parameters are allowed to be modified by users through the command line
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            # The parameters to be adjusted passed in during hyperparameter adjustment have a lower priority than the command line parameters.
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: Check the format of config file
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        # First load task config
        with open('./libcity/config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            # check model and dataset
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            # Modules that load dataset, executor, and evaluator
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = task_config[model]['dataset_class']
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                self.config['traj_encoder'] = task_config[model]['traj_encoder']
            if self.config['task'] == 'eta' and 'eta_encoder' not in self.config:
                self.config['eta_encoder'] = task_config[model]['eta_encoder']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            # For LSTM RNN GRU uses the same class, but the RNN module is different. Make some modifications here.
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
            # if self.config['dataset'] not in task_config['allowed_dataset']:
            #     raise ValueError('task {} do not support dataset {}'.format(
            #         self.config['task'], self.config['dataset']))
        # Then load the default config for each stage
        default_file_list = []
        # model
        default_file_list.append('model/{}/{}.json'.format(self.config['task'], self.config['model']))
        # dataset
        default_file_list.append('data/{}.json'.format(self.config['dataset_class']))
        # executor
        default_file_list.append('executor/{}.json'.format(self.config['executor']))
        # evaluator
        default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
        # Load all default configuration
        for file_name in default_file_list:
            with open('./libcity/config/{}'.format(file_name), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        # Model JSON is loaded before `data/<dataset_class>.json`. Model files often set
        # `add_time_in_day` / `add_day_in_week` / `load_external` to false, which blocks
        # the dataset-class file from applying (merge uses "only if key missing").
        # Re-apply those data-pipeline keys from the dataset-class JSON so
        # `TrafficStatePointDataset.json` (etc.) remains authoritative for fusion flags.
        _dataset_class_cfg_path = './libcity/config/data/{}.json'.format(
            self.config['dataset_class'])
        if os.path.exists(_dataset_class_cfg_path):
            with open(_dataset_class_cfg_path, 'r') as f:
                _ds_class = json.load(f)
            for _key in (
                    'add_time_in_day', 'add_day_in_week', 'load_external',
                    'normal_external', 'ext_scaler', 'add_kg_embeddings'):
                if _key in _ds_class:
                    self.config[_key] = _ds_class[_key]
        # Load data set config.json
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        # Gracefully fall back to CPU when CUDA is unavailable
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        else:
            use_gpu = False
            self.config['gpu'] = False
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # Supports iterative operations
    def __iter__(self):
        return self.config.__iter__()
