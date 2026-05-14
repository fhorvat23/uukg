"""Train and evaluate a single model."""
import argparse
import glob
import json
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import random

from libcity.config import ConfigParser
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args

log = './log/'


def _build_other_args(dict_args):
    return {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}


def _exp_id_from_evaluate_csv(csv_path):
    """Resolve libcity experiment id from .../libcity/cache/<exp_id>/evaluate_cache/..."""
    parts = os.path.normpath(csv_path).split(os.sep)
    for i, name in enumerate(parts):
        if name == 'cache' and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    raise ValueError('Cannot parse exp_id from path: {}'.format(csv_path))


def _libcity_log_for_exp(exp_id, model, dataset):
    """Log file name: <exp_id>-<model>-<dataset>-<Month-dd-YYYY_HH-MM-SS>.log"""
    pattern = os.path.join('./libcity/log', '{}-{}-{}-*.log'.format(exp_id, model, dataset))
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        return None
    return os.path.normpath(matches[-1])


if __name__ == '__main__':
    total = 5
    session_id = uuid.uuid4().hex[:12]
    started_at = datetime.now().isoformat(timespec='seconds')

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--kg_model', type=str, default=None,
                        help='KG embedding model name used to build node embeddings '
                             '(e.g. RefH, RotH, GIE). When set, the files '
                             '{CITY}_{kg_model}_region/poi/road_embeddings.npy are '
                             'fused into the input features. Leave unset to skip fusion.')
    parser.add_argument('--embed_dir', type=str, default=None,
                        help='Base directory that contains per-city KG embedding folders '
                             '(default: ../UrbanKG_Embedding_Model/UrbanKG_Embedding)')
    add_general_args(parser)
    base_args = parser.parse_args()
    base_dict = vars(base_args).copy()

    seeds_used = []
    seeds = [5593,
    2397,
    1220,
    4398,
    6196]
    # seeds = [
    # 4024,
    # 5680,
    # 1697,
    # 6026,
    # 5595,
    # 7169,
    # 7114,
    # 3823,
    # 5,
    # 4185
    # ]
    for i in range(total):
        seed = random.randint(0, 10000)
        seed = seeds[i]
        seeds_used.append(seed)
        dict_args = dict(base_dict)
        dict_args['seed'] = seed
        other_args = _build_other_args(dict_args)
        run_model(task=dict_args['task'], model_name=dict_args['model'],
                  dataset_name=dict_args['dataset'],
                  config_file=dict_args['config_file'],
                  saved_model=dict_args['saved_model'],
                  train=dict_args['train'], other_args=other_args)

    args = base_args
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

    finished_at = datetime.now().isoformat(timespec='seconds')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(save_path, 'sessions', '{}_{}'.format(stamp, session_id))
    os.makedirs(session_dir, exist_ok=True)

    horizon = np.arange(1, num_horizons + 1)
    df_avg = pd.DataFrame(avg_results, columns=metrics)
    df_avg.insert(0, 'horizon', horizon)
    df_avg.to_csv(os.path.join(session_dir, 'avg_result.csv'), index=False)

    df_arr = pd.DataFrame(arr_results, columns=metrics)
    df_arr.insert(0, 'horizon', horizon)
    df_arr.to_csv(os.path.join(session_dir, 'arr_result.csv'), index=False)

    runs = []
    for i, csv_path in enumerate(result_csv):
        eid = _exp_id_from_evaluate_csv(csv_path)
        runs.append({
            'run_index': i,
            'seed': seeds_used[i],
            'exp_id': eid,
            'cache_dir': os.path.normpath(os.path.join('./libcity/cache', str(eid))),
            'evaluate_csv': os.path.normpath(csv_path),
            'libcity_log': _libcity_log_for_exp(eid, args.model, args.dataset),
        })

    # Same merge order as training (CLI > config file > defaults + dataset-class overrides)
    _resolved_args = dict(base_dict)
    _resolved_args['seed'] = seeds_used[0]
    _resolved_other = _build_other_args(_resolved_args)
    resolved = ConfigParser(
        _resolved_args['task'],
        _resolved_args['model'],
        _resolved_args['dataset'],
        _resolved_args['config_file'],
        _resolved_args['saved_model'],
        _resolved_args['train'],
        _resolved_other,
    )
    rc = resolved.config
    _kg = rc.get('kg_model')
    _kg_used = _kg is not None and str(_kg).strip() != ''

    meta = {
        'session_id': session_id,
        'started_at': started_at,
        'finished_at': finished_at,
        'task': args.task,
        'model': args.model,
        'dataset': args.dataset,
        'config_file': args.config_file,
        'add_time_in_day': rc.get('add_time_in_day', False),
        'add_day_in_week': rc.get('add_day_in_week', False),
        'kg_embedding_model_used': _kg_used,
        'kg_model': _kg,
        'add_kg_embeddings': rc.get('add_kg_embeddings', True),
        'embed_dir': rc.get('embed_dir'),
        'total_runs': total,
        'seeds': seeds_used,
        'exp_ids': [r['exp_id'] for r in runs],
        'evaluate_csv_paths': [os.path.normpath(p) for p in result_csv],
        'runs': runs,
        'metrics': metrics,
        'libcity_log_naming': '{exp_id}-{model}-{dataset}-{Month-dd-YYYY_HH-MM-SS}.log',
    }
    with open(os.path.join(session_dir, 'session.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
