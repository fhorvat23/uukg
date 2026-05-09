"""Train and evaluate a single model."""
import argparse
import os
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args
import random
import pandas as pd
import numpy as np

log = './log/'

if __name__ == '__main__':
    total = 10
    for i in range(total):
        seed = random.randint(0, 10000)

        parser = argparse.ArgumentParser()
        # Add required CLI arguments
        parser.add_argument('--task', type=str,
                            default='traffic_state_pred', help='the name of task')
        parser.add_argument('--model', type=str,
                            default='RNN', help='the name of model')
        parser.add_argument('--dataset', type=str,
                            default='CHITaxi20190406', help='the name of dataset')
        parser.add_argument('--config_file', type=str,
                            default=None, help='the file name of config file')
        parser.add_argument('--saved_model', type=str2bool,
                            default=True, help='whether save the trained model')
        parser.add_argument('--train', type=str2bool, default=True,
                            help='whether re-train model if the model is trained before')
        parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
        parser.add_argument('--seed', type=int, default=seed, help='random seed')
        parser.add_argument('--kg_model', type=str, default=None,
                            help='KG embedding model name used to build node embeddings '
                                 '(e.g. RefH, RotH, GIE). When set, the files '
                                 '{CITY}_{kg_model}_region/poi/road_embeddings.npy are '
                                 'fused into the input features. Leave unset to skip fusion.')
        parser.add_argument('--embed_dir', type=str, default=None,
                            help='Base directory that contains per-city KG embedding folders '
                                 '(default: ../UrbanKG_Embedding_Model/UrbanKG_Embedding)')
        # Add other optional arguments
        add_general_args(parser)
        # Parse arguments
        args = parser.parse_args()
        dict_args = vars(args)
        other_args = {key: val for key, val in dict_args.items() if key not in [
            'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
            val is not None}
        run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
                  config_file=args.config_file, saved_model=args.saved_model,
                  train=args.train, other_args=other_args)

    save_path = log + args.dataset[0:3] + '/' + args.model + '_' + args.dataset + '/'

    os.makedirs(save_path, exist_ok=True)
    pattern = os.path.join('./libcity/cache', '*', 'evaluate_cache', '*_{}_{}.csv'.format(args.model, args.dataset))
    result_csv = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not result_csv:
        raise FileNotFoundError('No evaluation CSV found for model={} dataset={}.'.format(args.model, args.dataset))
    result_csv = result_csv[-total:]
    metrics = ["MAE", "RMSE", 'micro-F1', 'macro-F1']
    first_result = pd.read_csv(result_csv[0])
    num_horizons = first_result.shape[0]
    num_runs = len(result_csv)
    final_results_ten_train = np.zeros([num_horizons, len(metrics), num_runs])
    for i, csv_path in enumerate(result_csv):
        result = pd.read_csv(csv_path)
        temp = result[metrics].values
        final_results_ten_train[:, :, i] = temp

    avg_results = np.zeros([num_horizons, len(metrics)])
    arr_results = np.zeros([num_horizons, len(metrics)])
    ## Mean and variance of metrics
    for j in range(final_results_ten_train.shape[0]):
        avg_mae_rmse_microf1_macrof1 = np.mean(final_results_ten_train[j, :, :], axis=1)
        arr_mae = np.var(final_results_ten_train[j, :, :][0])
        arr_rmse = np.var(final_results_ten_train[j, :, :][1])
        arr_microf1 = np.var(final_results_ten_train[j, :, :][2])
        arr_macrof1 = np.var(final_results_ten_train[j, :, :][3])


        avg_results[j] = avg_mae_rmse_microf1_macrof1
        arr_results[j][0] = arr_mae
        arr_results[j][1] = arr_rmse
        arr_results[j][2] = arr_microf1
        arr_results[j][3] = arr_macrof1

    np.savetxt(save_path + 'avg_result.csv', avg_results)
    np.savetxt(save_path + 'arr_result.csv', arr_results)