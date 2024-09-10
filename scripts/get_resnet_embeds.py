import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.resnet import ResNet18D1D
from src.datasets import SpiroDataset


def get_embeds(data_path, model, device, batch_size, num_workers, pin_memory):
    dataset = SpiroDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    
    eids = []
    embeds = []
    labels = []
    for batch in tqdm(data_loader, desc='Creating data', total=len(data_loader)):
        with torch.no_grad():
            x = batch['flow_volume'].unsqueeze(1).to(device)
            embed = model(x)
        embeds.append(embed)
        labels.extend(batch['label'])
        eids.extend(batch['eid'])
        
    X = torch.cat(embeds, dim=0).cpu().numpy()
    y = np.array(labels)
    eids = np.array(eids)
    return X, y, eids


def main():
    data_dir = 'data/'
    train_data = f'{data_dir}/train_spiro.pkl'
    val_data = f'{data_dir}/val_spiro.pkl'
    test_data = f'{data_dir}/test_spiro.pkl'
    batch_size = 128
    num_workers = 20
    pin_memory = True
    ckpt_path = 'ckpts/AsthmaResNet/v0/best-epoch=5-step=9795.ckpt'

    # Load pretrained ResNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18D1D.from_lightning_checkpoint(ckpt_path).to(device).eval()

    # Generate spiro embeddings with pretrained ResNet
    X_train, y_train, eids_train = get_embeds(train_data, model, device, batch_size, num_workers, pin_memory)
    X_val, y_val, eids_val = get_embeds(val_data, model, device, batch_size, num_workers, pin_memory)
    X_test, y_test, eids_test = get_embeds(test_data, model, device, batch_size, num_workers, pin_memory)
    
    # Save data:
    with open(f'{data_dir}/train_spiro_embeds.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, eids_train), f)
    with open(f'{data_dir}/val_spiro_embeds.pkl', 'wb') as f:
        pickle.dump((X_val, y_val, eids_val), f)
    with open(f'{data_dir}/test_spiro_embeds.pkl', 'wb') as f:
        pickle.dump((X_test, y_test, eids_test), f)


if __name__ == '__main__':
    main()