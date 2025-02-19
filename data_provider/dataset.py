import os
import pandas as pd
import sys
import pickle
import random
from pathlib import Path

import lmdb
import numpy as np
import scanpy as sc
import torch
from torch.utils.data.dataset import Dataset
import scipy.sparse

from common import config as cfg


class LMDBReader:
    def __init__(self, db_path):
        self._envs = {}
        for x in list(db_path.glob("*")):
            k = x.name
            self._envs[k] = lmdb.open(
                str(x),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def get_file_names(self):
        return list(self._envs.keys())

    def get_hca_shape(self, name):
        with self._envs[name].begin(write=False) as txn:
            v = txn.get(f"shape".encode("ascii"))
            return pickle.loads(v)

    def get_item(self, name, row):
        with self._envs[name].begin(write=False) as txn:
            v = txn.get(f"item:{row}".encode("ascii"))
            return pickle.loads(v)

    def get_gene(self, name, cols):
        with self._envs[name].begin(write=False) as txn:
            v = []
            for col in cols:
                x = txn.get(f"gene:{col}".encode("ascii")).decode("ascii")
                v.append(x)
            return v


class H5ADReader:
    def __init__(self, path_list):
        self.path_list = path_list
        self.data = {}
        for k in path_list:
            self.data[k] = {}
            hca_temp = sc.read_h5ad(k)
            n_cells = hca_temp.n_obs
            shuffled_indices = np.random.permutation(n_cells)
            hca = hca_temp[shuffled_indices].copy()

            hca.obs["index"] = range(len(hca.obs))
            self.data[k]["hca"] = hca
            self.data[k]["perturb_rows"] = hca.obs.loc[
                (hca.obs["perturbation"] != 'control')
            ]["index"].tolist()
            self.data[k]["control_rows"] = hca.obs.loc[
                (hca.obs["perturbation"] == 'control')
            ]["index"].tolist()


    def get_file_names(self):
        return self.path_list

    def get_hca_shape(self, name):
        hca = self.data[name]["hca"]
        return hca.shape

    def get_item(self, name, row):
        hca = self.data[name]["hca"]
        return hca.X[row]

    def get_gene(self, name, cols):
        hca = self.data[name]["hca"]
        v = []
        for col in cols:
            v.append(hca.var.index[col])
        return v

    def get_perturb_rows(self, name):
        return self.data[name]["perturb_rows"]

    def get_control_rows(self, name):
        return self.data[name]["control_rows"]

    def get_perturbation(self, name, row):
        hca = self.data[name]["hca"]
        hca.obs['perturbation'] = pd.Categorical(hca.obs['perturbation'])
        return hca.obs.perturbation.cat.codes.iloc[row]


class HCADataset(Dataset):
    def __init__(self, path_list, train=True):
        self.train = train
        self.reader = H5ADReader(path_list)
        self.hca = self.reader.get_file_names()
        self.weights = []
        for name in self.hca:
            self.weights.append(self.reader.get_hca_shape(name)[0])
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        self.perturb_all = self.reader.data[self.hca[0]]['hca'].obs.perturbation
        self.gene2token = {}
        
        for i, x in enumerate(open(cfg.pretrain_model_dir / "gene_name.txt")):
            self.gene2token[x.strip()] = i
        

        for k in range(len(self.hca)):
            for i, x in enumerate(list(self.reader.data[self.hca[k]]['hca'].var.index)):
                if x.strip() not in self.gene2token:
                    self.gene2token[x.strip()] = i

        self.perturb_rows = {}
        for k in self.reader.get_file_names():
            perturb_rows = self.reader.get_perturb_rows(k)
            if self.train:
                self.perturb_rows[k] = [_ for _ in perturb_rows if _ % 10 != 0]
                # self.perturb_rows[k] = [_ for _ in perturb_rows]
            else:
                self.perturb_rows[k] = [_ for _ in perturb_rows if _ % 20 == 0]
                # self.perturb_rows[k] = [_ for _ in perturb_rows]
            self.cell_name = self.reader.data[self.hca[0]]['hca'].obs.index[self.perturb_rows[k]]

            print(f"dataset info: {k} {len(self.perturb_rows[k])}")

    def __len__(self):
        return sum([len(_) for k, _ in self.perturb_rows.items()])

    def downsample_v1(self, a, p):
        eps = 1e-12
        umi = a.sum() * p
        b = a * umi / (a.sum() + eps) + np.random.random(a.shape)
        b = np.floor(b).astype(int)
        return b

    def downsample_v2(self, a, p):
        rnd = scipy.stats.binom(a, p)
        return rnd.rvs()

    def get_sample(self, hca, row, aug_prob, ratio):
        hca_shape = self.reader.get_hca_shape(hca)
        perturb = self.reader.get_perturbation(hca, row)

        rawcount = self.reader.get_item(hca, row)
    
        if not isinstance(rawcount, np.ndarray):
            rawcount = rawcount.toarray()
        rawcount = np.squeeze(rawcount.astype(int))

        if self.train:
            rawcount = self.downsample_v2(rawcount, aug_prob)

        ds = rawcount
        
        # ds = self.downsample_v2(rawcount, 1.0 / ratio)

        u_cols = np.arange(ds.shape[0])[ds > 0]

        if self.train and u_cols.shape[0] > 2048:
            u_cols = np.random.choice(u_cols, 2048, replace=False)
        u_genes = self.reader.get_gene(hca, u_cols)
        
        def normalize(x):
            eps = 1e-12
            x = np.log1p(x / (x.sum(axis=0) + eps) * 10000)
            return x

        ds_norm = normalize(ds)
        # ds_norm = ds

        u_token = []

        for x in self.reader.get_gene(hca, u_cols):
            u_token.append(self.gene2token[x])
        
        return {
            "u_token": u_token,
            "ds": ds[u_cols],
            "ds_norm": ds_norm[u_cols],
            "perturb": np.array([perturb]),
            "index": u_cols,
            "genes": u_genes
        }

    def __getitem__(self, index):
        hca = np.random.choice(self.hca, p=self.weights) if self.train else self.hca[0]
        aug_prob = np.random.random() * 0.9 + 0.1
        ratio = 5

        data = []
        mx_len = 0

        # mx_token = 16 * 256 * 16
        mx_token = 16 * 256

        for i in range(1024 if self.train else 1):
            row = None
            if self.train:
                row = np.random.choice(self.perturb_rows[hca])
            else:
                row = self.perturb_rows[hca][index]
            sample = self.get_sample(hca, row, aug_prob, ratio)
            cur_len = len(sample["u_token"])
            if len(data) > 0 and max(mx_len, cur_len) * (len(data) + 1) > mx_token: # to ensure one iteration has enough data to train
                break
            data.append(sample)
            mx_len = max(mx_len, cur_len)
    


        def _pad_fn(a, value):
            a = [np.array(_) for _ in a]
            max_shape = np.max([_.shape for _ in a], axis=0)
            na = []
            for x in a:
                pad_shape = [(0, l2 - l1) for l1, l2 in zip(x.shape, max_shape)]
                na.append(np.pad(x, pad_shape, mode="constant", constant_values=value))
            return np.stack(na)

        pad_values = {
            "u_token": 40000,
            "ds": 0,
            "ds_norm": 0.0,
            "perturb": -1,
            "index": 0,
            "genes": "padding"
        }

        ret = {}
        ret["ratio"] = np.array([ratio])
        for key in data[0].keys():
            ret[key] = _pad_fn([_[key] for _ in data], pad_values[key])
                
        return ret


class TrainDataset(Dataset):
    def __init__(self):
        ds = HCADataset(
            # [os.path.join(cfg.dataset_dir, "GSE197268_scleap.h5ad")]
            # [os.path.join(cfg.dataset_dir, "GSE197268_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "Lymphoma_8k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "GSE197268_intersect.h5ad")],
            # [os.path.join(cfg.dataset_dir, "test_rest_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "9268_rest_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "lym_rest_5k_2.h5ad")],
            # [os.path.join(cfg.dataset_dir, "li_rest_5k_2.h5ad")],
            # [os.path.join(cfg.dataset_dir, "GSE262072_filtered_5k.h5ad")],
            [os.path.join(cfg.dataset_dir, "temp_train_ad.h5ad")],
        )
        self._dataset = ds
        self._cell_name = ds.cell_name
        self._perturb_all = ds.perturb_all


    def __getitem__(self, index):
        data = self._dataset[index]
        print(data["genes"].shape)
        if os.path.exists(cfg.saved_dir / 'train_input_genes.csv'):
            pd.DataFrame(data["genes"]).to_csv(cfg.saved_dir / 'train_input_genes_temp.csv', index=False, header=False)
        else:
            pd.DataFrame(data["genes"]).to_csv(cfg.saved_dir / 'train_input_genes.csv', index=False, header=False)
            
        if os.path.exists(cfg.saved_dir / 'train_label.csv'):  
            pd.DataFrame(data["perturb"][:, 0]).to_csv(cfg.saved_dir / 'train_label_temp.csv', index=False, header=False)
        else:
            pd.DataFrame(data["perturb"][:, 0]).to_csv(cfg.saved_dir / 'train_label.csv', index=False, header=False)
        print(data["perturb"][:, 0])
        return {
            "ratio": torch.tensor(data["ratio"]).long(),
            "u_token": torch.tensor(data["u_token"]).long(),
            "ds": torch.tensor(data["ds"]).long(),
            "ds_norm": torch.tensor(data["ds_norm"]).float(),
            "perturb": torch.tensor(data["perturb"]).long()[:, 0]
        }


class ValidationDataset(Dataset):
    def __init__(self):
        ds = HCADataset(
            # [os.path.join(cfg.dataset_dir, "GSE197268_scleap.h5ad")],
            # [os.path.join(cfg.dataset_dir, "Lymphora_scleap.h5ad")],
            # [os.path.join(cfg.dataset_dir, "GSE150992_intersect_lym_sample.h5ad")],
            # [os.path.join(cfg.dataset_dir, "Lymphoma_8k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "GSE197268_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "test_1234_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "9268_1234_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "lym_rest_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "li_1234_5k.h5ad")],
            # [os.path.join(cfg.dataset_dir, "GSE262072_filtered_5k.h5ad")],
            [os.path.join(cfg.dataset_dir, "temp_test_ad.h5ad")],
            train=False,
        )
        self._dataset = ds
        self._cell_name = ds.cell_name
        self._perturb_all = ds.perturb_all


    def __getitem__(self, index):
        data = self._dataset[index]
        # pd.DataFrame(data["genes"]).to_csv('val_input_genes.csv', index=False, header=False)
        # loaded_matrix = pd.read_csv('matrix.csv', header=None).values
        return {
            "ratio": torch.tensor(data["ratio"]).long(),
            "u_token": torch.tensor(data["u_token"]).long(),
            "ds": torch.tensor(data["ds"]).long(),
            "ds_norm": torch.tensor(data["ds_norm"]).float(),
            "perturb": torch.tensor(data["perturb"]).long()[:, 0]
        }

    def __len__(self):
        return len(self._dataset)