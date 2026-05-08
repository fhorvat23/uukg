from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, get_executor, get_logger, set_random_seed
import random

"""Take one batch of data for preliminary model testing."""

# Load configuration file
config = ConfigParser(task='traffic_state_pred', model='RNN',
                      dataset='METR_LA', other_args={'batch_size': 2})
exp_id = config.get('exp_id', None)
if exp_id is None:
    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = exp_id
# logger
logger = get_logger(config)
logger.info(config.config)
# seed
seed = config.get('seed', 0)
set_random_seed(seed)
# Load data module
dataset = get_dataset(config)
# Data preprocessing, dividing data sets
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
# Extract a batch of data for model testing
batch = train_data.__iter__().__next__()
# Load model
model = get_model(config, data_feature)
model = model.to(config['device'])
# Load executor
executor = get_executor(config, model, data_feature)
# Model prediction
batch.to_tensor(config['device'])
res = model.predict(batch)
logger.info('Result shape is {}'.format(res.shape))
logger.info('Success test the model!')
