import torch
import pandas as pd
import lightning as L
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from .tokenizer import PhenotypeTokenizer


class MHMDataset(Dataset):
    def __init__(self, df, tokenizer, mhm_probability=0.15):
        self.eids = []
        self.data = []
        self.tokenizer = tokenizer
        self.mhm_probability = mhm_probability
        self._tokenize(df)
    
    def _tokenize(self, df):
        for _, row in tqdm(df.iterrows(), desc='Tokenize data', total=len(df)):
            row = row.to_dict()
            if 'eid' in row:
                eid = row.pop('eid')
                self.eids.append(eid)
            self.data.append(self.tokenizer.encode(row))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        # Create value_ids, phenotype_ids and labels for MHM and boolean trait prediction
        eid = None
        if self.eids:
            eid = self.eids[idx]
        phenotype_ids = torch.tensor(self.data[idx]['phenotype_ids'])
        hm_value_ids = torch.tensor(self.data[idx]['value_ids'])
        pred_value_ids = torch.tensor(self.data[idx]['value_ids'])
        hm_labels = hm_value_ids.clone()
        pred_labels = pred_value_ids.clone()
        
        # Create probability matrix for masking
        prob_matrix = torch.full(hm_labels.shape, self.mhm_probability)
                
        # Create mask for health modeling tokens prediction
        hm_mask = torch.bernoulli(prob_matrix).bool()
        hm_labels[~hm_mask] = -100    # only compute loss on masked tokens
        hm_value_ids[hm_mask] = self.tokenizer.mask_token_id
        
        # Creat mask for boolean trait prediction
        bool_trait_ids = torch.tensor(list(self.tokenizer.boolean_traits.keys()))
        pred_mask = (phenotype_ids[..., None] == bool_trait_ids).any(dim=-1)
        pred_value_ids[pred_mask] = self.tokenizer.mask_token_id
        
        return {
            'phenotype_ids': phenotype_ids,
            'hm_value_ids': hm_value_ids,
            'hm_labels': hm_labels,
            'pred_value_ids': pred_value_ids,
            'pred_labels': pred_labels,
            'eid': eid
        }


class MHMDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, val_data: str, test_data: str,
                 num_features: List[str], cat_features: List[str],
                 batch_size: int = 32, n_workers: int = 4,
                 n_bins: int = 100, binning: str = 'uniform',
                 mhm_probability: float = 0.15,
                 pin_memory: bool = True):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_features = num_features
        self.cat_features = cat_features
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_bins = n_bins
        self.binning = binning
        self.mhm_probability = mhm_probability
        self.pin_memory = pin_memory
        self.tokenizer = PhenotypeTokenizer(n_bins=n_bins, binning=binning)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._has_setup = False

    def setup(self, stage=None):
        if not self._has_setup and (stage == 'fit' or stage is None):
            train_df = pd.read_csv(self.train_data)
            val_df = pd.read_csv(self.val_data)
            train_df = train_df[['eid'] + self.num_features + self.cat_features]
            val_df = val_df[['eid'] + self.num_features + self.cat_features]
            self.tokenizer.fit(pd.concat([train_df, val_df]), self.num_features, self.cat_features)
            
            self.train_dataset = MHMDataset(train_df, self.tokenizer, mhm_probability=self.mhm_probability)
            self.val_dataset = MHMDataset(val_df, self.tokenizer, mhm_probability=self.mhm_probability)
            self._has_setup = True
        
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_data)
            test_df = test_df[['eid'] + self.num_features + self.cat_features]
            self.test_dataset = MHMDataset(test_df, self.tokenizer, mhm_probability=self.mhm_probability)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )


class SpiroDataset(Dataset):
    def __init__(self, data_path, balance=False):
        self.data = pd.read_pickle(data_path)
        if balance:
            self.data = self._balance_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'eid': int(self.data.iloc[idx]['eid']),
            'flow_volume': torch.from_numpy(self.data.iloc[idx]['flow_volume']),
            'label': int(self.data.iloc[idx]['label']),
        }

    def _balance_dataset(self):
        labels = self.data['label']
        positive_samples = self.data[labels == 1]
        negative_samples = self.data[labels == 0]
        
        # Undersample the majority class
        if len(positive_samples) < len(negative_samples):
            negative_samples = negative_samples.sample(n=len(positive_samples), random_state=42)
        else:
            positive_samples = positive_samples.sample(n=len(negative_samples), random_state=42)
        
        # Combine the balanced datasets
        balanced_df = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        return balanced_df

    def get_prevalence(self):
        labels = self.data['label']
        total_samples = len(labels)
        positive_samples = sum(labels)
        negative_samples = total_samples - positive_samples
        
        return {
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_ratio": positive_samples / total_samples,
            "negative_ratio": negative_samples / total_samples
        }

class SpiroDataModule(L.LightningDataModule):
    def __init__(self, train_data: str, val_data: str, test_data: str,
                 batch_size: int = 32, n_workers: int = 4,
                 pin_memory: bool = True, balance_train: bool = False):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.balance_train = balance_train
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SpiroDataset(self.train_data, balance=self.balance_train)
            self.val_dataset = SpiroDataset(self.val_data)
            print("Train dataset prevalence:")
            self._print_prevalence(self.train_dataset)
            print("Validation dataset prevalence:")
            self._print_prevalence(self.val_dataset)

        if stage == 'test' or stage is None:
            self.test_dataset = SpiroDataset(self.test_data)
            print("Test dataset prevalence:")
            self._print_prevalence(self.test_dataset)

    def _print_prevalence(self, dataset):
        prevalence = dataset.get_prevalence()
        print(f"Total samples: {prevalence['total_samples']}")
        print(f"Positive samples (cases): {prevalence['positive_samples']} ({prevalence['positive_ratio']:.2%})")
        print(f"Negative samples (controls): {prevalence['negative_samples']} ({prevalence['negative_ratio']:.2%})")
        print()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory
        )


class TabSpiroDataset(Dataset):
    def __init__(self, tab_path, spiro_path, features, tokenizer):
        self.eids = []
        self.tab_data = []
        self.spiro_data = []
        self.labels = []
        self.tokenizer = tokenizer
        
        df_tab = pd.read_csv(tab_path)[['eid'] + features + ['Asthma']]
        df_spiro = pd.read_pickle(spiro_path)
        df = pd.merge(df_tab, df_spiro, on='eid')
        assert len(df) == len(df_spiro)
        self._tokenize(df)
    
    def _tokenize(self, df):
        for _, row in tqdm(df.iterrows(), desc='Tokenize data', total=len(df)):
            row = row.to_dict()
            eid = row.pop('eid')
            label = row.pop('label')
            flow_volume = row.pop('flow_volume')
            self.eids.append(eid)
            self.labels.append(label)
            self.spiro_data.append(flow_volume)
            self.tab_data.append(self.tokenizer.encode(row))
        
    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):        
        eid = self.eids[idx]
        label = self.labels[idx]
        phenotype_ids = torch.tensor(self.tab_data[idx]['phenotype_ids'])
        value_ids = torch.tensor(self.tab_data[idx]['value_ids'])
        flow_volume = torch.from_numpy(self.spiro_data[idx])
        
        # Mask asthma:
        p_id = self.tokenizer.get_phenotype_id('Asthma')
        mask = phenotype_ids == p_id
        label_ids = value_ids[mask].squeeze()
        value_ids[mask] = self.tokenizer.mask_token_id

        return {
            'eid': eid,
            'phenotype_ids': phenotype_ids,
            'value_ids': value_ids,
            'flow_volume': flow_volume,
            'mask': mask,
            'labels': label_ids,
            'label': int(label)
        }


class TabSpiroDataModule(L.LightningDataModule):
    def __init__(self, tab_train: str, tab_val: str, tab_test: str,
                 spiro_train: str, spiro_val: str, spiro_test: str,
                 ckpt_tokenizer: str,
                 num_features: List[str], cat_features: List[str],
                 batch_size: int = 32, n_workers: int = 4,
                 pin_memory: bool = True):
        super().__init__()
        self.tab_train = tab_train
        self.tab_val = tab_val
        self.tab_test = tab_test
        self.spiro_train = spiro_train
        self.spiro_val = spiro_val
        self.spiro_test = spiro_test
        self.num_features = num_features
        self.cat_features = cat_features
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.tokenizer = torch.load(ckpt_tokenizer, map_location=torch.device('cpu'))['hyper_parameters']['tokenizer']
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._has_setup = False

    def setup(self, stage=None):
        if not self._has_setup and (stage == 'fit' or stage is None):
            self.train_dataset = TabSpiroDataset(self.tab_train, self.spiro_train, self.num_features + self.cat_features, self.tokenizer)
            self.val_dataset = TabSpiroDataset(self.tab_val, self.spiro_val, self.num_features + self.cat_features, self.tokenizer)
            self._has_setup = True
        
        if stage == 'test' or stage is None:
            self.test_dataset = TabSpiroDataset(self.tab_test, self.spiro_test, self.num_features + self.cat_features, self.tokenizer)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.n_workers,
            pin_memory = self.pin_memory
        )