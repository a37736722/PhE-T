import argparse
import torch
import lightning as L
from importlib.machinery import SourceFileLoader
from src.datasets import MHMDataModule, TabSpiroDataModule
from src.models import MHMPhET, AsthmaPhET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of the model to test. Should be either 'phet' or 'as-phet'.", type=str, required=True)
    parser.add_argument("--ckpt_path", help="Path to a checkpoint to resume from.", type=str, required=True)
    parser.add_argument("--out_dir", help="Path to the save the prediction", type=str, required=True)
    parser.add_argument("--config", help="Path to the config file.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=1)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    # Parse arguments:
    args = parse_args()
    cfg = SourceFileLoader("config", args.config).load_module()
    model = args.model
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    out_dir = args.out_dir
    
    # Create data module:
    if model == 'phet':
        dm = MHMDataModule(
            n_workers = nb_workers,
            pin_memory = False,
            **cfg.data_module_cfg
        )
        dm.setup('fit')
    elif model == 'as-phet':
        dm = TabSpiroDataModule(
            n_workers = nb_workers,
            pin_memory = False,
            **cfg.data_module_cfg
        )
    
    # Create model:
    if model == 'phet':
        model = MHMPhET(
            tokenizer = dm.tokenizer,
            out_dir = out_dir,
            **cfg.model_cfg
        )
    elif model == 'as-phet':
        model = AsthmaPhET(
            out_dir = out_dir,
            **cfg.model_cfg
        )
    
    # Set trainer:
    trainer = L.Trainer(
        accelerator = 'cpu',
        logger = False,
    )

    # Test model:
    trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()