import argparse
import json
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
DATA_PATH = './data'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="NYC", choices=["NYC", "CHI"],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="GIE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                       '"ComplEx", "RotatE",'
                                                       '"RotH", "RefH", "AttH"'
                                                       '"GIE'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=4120, type=int, help="Batch size"
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--checkpoint_path",
    default="./logs/05_05/NYC/RefH_22_06_01/model.pt",
    type=str,
    help="Path to trained checkpoint model.pt"
)


def load_entity2id_map(entity2id_path):
    entity2kg = {}
    with open(entity2id_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            entity_key, kg_id = parts
            entity2kg[entity_key] = int(kg_id)
    return entity2kg


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of columns {candidates} found in dataframe columns: {list(df.columns)}")


def build_kg_mapping_files(dataset):
    processed_dir = os.path.join("../UrbanKG_data", "Processed_data", dataset)
    urbankg_dir = os.path.join("../UrbanKG_data", "UrbanKG", dataset)

    entity2id_path = os.path.join(urbankg_dir, f"entity2id_{dataset}.txt")
    poi_path = os.path.join(processed_dir, f"{dataset}_poi.csv")
    road_path = os.path.join(processed_dir, f"{dataset}_road.csv")
    area_path = os.path.join(processed_dir, f"{dataset}_area.csv")

    entity2kg = load_entity2id_map(entity2id_path)

    poi_df = pd.read_csv(poi_path)
    road_df = pd.read_csv(road_path)
    area_df = pd.read_csv(area_path)

    poi_id_col = first_existing_column(poi_df, ["poi_id", "POI_id", "id"])
    poi_region_col = first_existing_column(poi_df, ["area_id", "Region_id", "region_id"])
    road_id_col = first_existing_column(road_df, ["road_id", "link_id", "id"])
    road_region_col = first_existing_column(road_df, ["area_id", "Region_id", "region_id"])
    region_id_col = first_existing_column(area_df, ["region_id", "area_id", "OBJECTID", "Unnamed: 0"])

    poi_map_df = pd.DataFrame({
        "poi_id": poi_df[poi_id_col].astype(int),
        "KG_id": poi_df[poi_id_col].apply(lambda x: entity2kg.get(f"POI/{int(x)}")),
        "Region_id": poi_df[poi_region_col].astype(int),
    }).dropna(subset=["KG_id"])
    poi_map_df["KG_id"] = poi_map_df["KG_id"].astype(int)

    road_map_df = pd.DataFrame({
        "road_id": road_df[road_id_col].astype(int),
        "KG_id": road_df[road_id_col].apply(lambda x: entity2kg.get(f"Road/{int(x)}")),
        "Region_id": road_df[road_region_col].astype(int),
    }).dropna(subset=["KG_id"])
    road_map_df["KG_id"] = road_map_df["KG_id"].astype(int)

    region_map_df = pd.DataFrame({
        "region_id": area_df[region_id_col].astype(int),
    })
    region_map_df["KG_id"] = region_map_df["region_id"].apply(lambda x: entity2kg.get(f"Area/{int(x)}"))
    region_map_df = region_map_df.dropna(subset=["KG_id"])
    region_map_df["KG_id"] = region_map_df["KG_id"].astype(int)

    region_map_csv = os.path.join(processed_dir, f"{dataset}_area_KG_id.csv")
    poi_map_csv = os.path.join(processed_dir, f"{dataset}_poi_KG_id.csv")
    road_map_csv = os.path.join(processed_dir, f"{dataset}_road_KG_id.csv")

    region_map_df.to_csv(region_map_csv, index=False)
    poi_map_df.to_csv(poi_map_csv, index=False)
    road_map_df.to_csv(road_map_csv, index=False)

    print(f"Saved region mapping: {region_map_csv} ({len(region_map_df)} rows)")
    print(f"Saved poi mapping: {poi_map_csv} ({len(poi_map_df)} rows)")
    print(f"Saved road mapping: {road_map_csv} ({len(road_map_df)} rows)")

    return region_map_csv, poi_map_csv, road_map_csv


def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    config_path = os.path.join(os.path.dirname(args.checkpoint_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            ckpt_cfg = json.load(f)
        # Align inference model with training config to avoid state_dict mismatch.
        args.model = ckpt_cfg.get("model", args.model)
        args.rank = int(ckpt_cfg.get("rank", args.rank))
        args.multi_c = bool(ckpt_cfg.get("multi_c", args.multi_c))
        args.bias = ckpt_cfg.get("bias", args.bias)
        args.dtype = ckpt_cfg.get("dtype", args.dtype)
        print(
            f"Loaded checkpoint config from {config_path}: "
            f"model={args.model}, rank={args.rank}, multi_c={args.multi_c}"
        )
    else:
        print(f"Warning: config.json not found next to checkpoint: {config_path}")

    model = getattr(models, args.model)(args)
    checkpoint_state = torch.load(args.checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint_state)
    except RuntimeError as e:
        raise RuntimeError(
            "Checkpoint/model mismatch. Ensure --checkpoint_path points to a checkpoint "
            "trained with the same model settings, or keep config.json beside model.pt.\n"
            f"Checkpoint: {args.checkpoint_path}\n"
            f"Current args: model={args.model}, rank={args.rank}, multi_c={args.multi_c}\n"
            f"Original error: {e}"
        ) from e
    entity_embeddings = model.entity.weight.detach().numpy()

    # File path: data/NYC/entity_idx_embedding.csv <- rename or change code
    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx_embedding.csv", header=None)
    entity_idx = np.array(idx)

    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for i in range(entity_embeddings.shape[0]):

        entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]


    return entity_final_embedddings


def get_region_embeddings(grid_KG_id_path, entity_final_embedddings, save_path):

    grid = pd.read_csv(grid_KG_id_path)
    grid_KG_id = grid[["region_id", "KG_id"]].values
    grid_embeddings = np.zeros([grid_KG_id.shape[0], entity_final_embedddings.shape[1]])

    for i in range(grid_embeddings.shape[0]):
        grid_embeddings[i] = entity_final_embedddings[int(grid_KG_id[i][1])]

    print(grid_embeddings)
    np.save(save_path, grid_embeddings)

def get_POI_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    poi = pd.read_csv(grid_KG_id_path)
    poi_KG_id = poi[["poi_id", "KG_id", "Region_id"]].values
    poi_embeddings = np.zeros([poi_KG_id.shape[0], entity_final_embedddings.shape[1] + 1])
    for i in range(poi_embeddings.shape[0]):
        poi_embeddings[i][0:entity_final_embedddings.shape[1]] = entity_final_embedddings[int(poi_KG_id[i][1])]
        poi_embeddings[i][entity_final_embedddings.shape[1]] = int(poi_KG_id[i][2])

    print(poi_embeddings)
    np.save(save_path, poi_embeddings)

def get_Road_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    road = pd.read_csv(grid_KG_id_path)
    road_KG_id = road[["road_id", "KG_id", "Region_id"]].values
    road_embeddings = np.zeros([road_KG_id.shape[0], entity_final_embedddings.shape[1] + 1])
    for i in range(road_embeddings.shape[0]):
        road_embeddings[i][0:entity_final_embedddings.shape[1]] = entity_final_embedddings[int(road_KG_id[i][1])]
        road_embeddings[i][entity_final_embedddings.shape[1]] = int(road_KG_id[i][2])

    print(road_embeddings)
    np.save(save_path, road_embeddings)


args = parser.parse_args()
entity_final_embedddings = get_embeddings(args)

embedding_save_dir = os.path.join("./UrbanKG_Embedding", args.dataset, f"{args.model}")
os.makedirs(embedding_save_dir, exist_ok=True)

region_map_csv, poi_map_csv, road_map_csv = build_kg_mapping_files(args.dataset)

prefix = f"{args.dataset}_{args.model}_"
region_embedding_npy = os.path.join(embedding_save_dir, f"{prefix}region_embeddings.npy")
poi_embedding_npy    = os.path.join(embedding_save_dir, f"{prefix}poi_embeddings.npy")
road_embedding_npy   = os.path.join(embedding_save_dir, f"{prefix}road_embeddings.npy")

get_region_embeddings(region_map_csv, entity_final_embedddings, region_embedding_npy)
get_POI_embedding(poi_map_csv, entity_final_embedddings, poi_embedding_npy)
get_Road_embedding(road_map_csv, entity_final_embedddings, road_embedding_npy)















