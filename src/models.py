import os
import json
import torch
import lightning as L
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict
from torch.optim import AdamW
from torcheval.metrics.functional import binary_auroc, binary_auprc
from transformers import get_linear_schedule_with_warmup
from .phet import PhET, PhETConfig
from .resnet import ResNet18D1D


class MHMPhET(L.LightningModule):
    def __init__(
        self,
        tokenizer,
        phet_config,
        learning_rate: float = 1e-4,
        adamw_epsilon: float = 1e-8,
        adamw_betas: tuple = (0.9, 0.98),
        warmup_steps: int = 10000,
        weight_decay: float = 0.0,
        out_dir: str = 'scores/phet'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        phet_config['p_size'] = tokenizer.p_size
        phet_config['v_size'] = tokenizer.v_size
        config = PhETConfig()
        config.update(**phet_config)
        phet = PhET(config)
        self.model = phet
        self.out_dir = out_dir
        
        self.tokenizer = tokenizer
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(
            value_ids = batch['hm_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            labels = batch['hm_labels'],
        )
        loss = outputs['loss']
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            value_ids = batch['hm_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            labels = batch['hm_labels'],
        )
        loss = outputs['loss']
        
        scores = self.model.predict(
            value_ids = batch['pred_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            bool_traits = self.tokenizer.boolean_traits
        )
        
        self.validation_step_outputs.append({
            'loss': loss,
            'scores': scores,
            'labels': batch['pred_labels'],
            'phenotype_ids': batch['phenotype_ids']
        })
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.tensor([x['loss'] for x in outputs], device=self.device).mean()
        y = defaultdict(lambda: defaultdict(list))
        for x in outputs:
            for trait, y_pred in x['scores'].items():
                p_id = self.tokenizer.get_phenotype_id(trait)
                info = self.tokenizer.get_boolean_trait_info(p_id)
                true_id = info['true_id']
                false_id = info['false_id']
                
                phenotype_ids = x['phenotype_ids']
                labels = x['labels'][phenotype_ids == p_id]
                y_true = labels.clone()
                y_true[labels == false_id] = 0
                y_true[labels == true_id] = 1
                y[trait]['y_pred'].append(y_pred)
                y[trait]['y_true'].append(y_true)
        
        results = {}
        for trait, value in y.items():
            pred_values = torch.cat(value['y_pred'])
            true_values = torch.cat(value['y_true'])
            
            # Compute AUROC and AUPRC
            auroc = binary_auroc(pred_values, true_values)
            auprc = binary_auprc(pred_values, true_values)
            results[trait] = {"AUROC": auroc, "AUPRC": auprc}

        # Log results
        self.log("val/loss", loss, sync_dist=True)
        for name, metrics in results.items():
            self.log(f"val/auroc/{name}", metrics['AUROC'], sync_dist=True)
            self.log(f"val/auprc/{name}", metrics['AUPRC'], sync_dist=True)

        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        scores = self.model.predict(
            value_ids = batch['pred_value_ids'],
            phenotype_ids = batch['phenotype_ids'],
            bool_traits = self.tokenizer.boolean_traits
        )
        
        self.test_step_outputs.append({
            'eids': batch['eid'],
            'scores': scores,
            'labels': batch['pred_labels'],
            'phenotype_ids': batch['phenotype_ids']
        })

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y = defaultdict(lambda: defaultdict(list))
        for x in outputs:
            eids = x['eids']
            for trait, y_scores in x['scores'].items():
                p_id = self.tokenizer.get_phenotype_id(trait)
                info = self.tokenizer.get_boolean_trait_info(p_id)
                true_id = info['true_id']
                false_id = info['false_id']
                
                phenotype_ids = x['phenotype_ids']
                labels = x['labels'][phenotype_ids == p_id]
                y_true = labels.clone()
                y_true[labels == false_id] = 0
                y_true[labels == true_id] = 1
                y[trait]['eids'].append(eids)
                y[trait]['y_scores'].append(y_scores)
                y[trait]['y_true'].append(y_true)

        # Save output in JSON file:
        os.makedirs(self.out_dir, exist_ok=True)
        for trait, value in y.items():
            eids = torch.cat(value['eids'])
            y_true = torch.cat(value['y_true'])
            y_scores = torch.cat(value['y_scores'])
                
            trait_name = trait.lower().replace(" ", "-")
            filename = f"rs_{trait_name}"
            filename += ".json"
            full_path = os.path.join(self.out_dir, filename)

            with open(full_path, "w") as f:
                f.write(json.dumps({
                    "eids": eids.tolist(),
                    "y_true": y_true.tolist(),
                    "y_scores": y_scores.tolist()
                }))
        print(f"Scores saved successfully in {self.out_dir}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adamw_epsilon, betas=self.hparams.adamw_betas)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class AsthmaResNet(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
        warmup_steps: int = 0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = ResNet18D1D()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.validation_step_outputs = []
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x = batch['flow_volume'].unsqueeze(1)
        y = batch['label'].float()
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['flow_volume'].unsqueeze(1)
        y = batch['label'].float()
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.validation_step_outputs.append({'loss': loss, 'y': y, 'y_hat': y_hat})
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        losses = torch.stack([x['loss'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        loss = torch.mean(losses)
        auroc = binary_auroc(y_hat, y.int())
        auprc = binary_auprc(y_hat, y.int())

        # Log results
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/auroc", auroc, sync_dist=True)
        self.log("val/auprc", auprc, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class AsthmaPhET(L.LightningModule):
    def __init__(self, ckpt_phet, ckpt_resnet, n_embeds=1, freeze_phet=True, freeze_resnet=True,
            learning_rate: float = 1e-4,
            weight_decay: float = 0,
            warmup_steps: int = 0,
            out_dir: str = 'scores/as-phet'
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['phet', 'resnet'])
        
        # Load checkpoints
        checkpoint = torch.load(ckpt_phet, map_location=torch.device('cpu'))
        self.tokenizer = checkpoint['hyper_parameters']['tokenizer']
        self.phet = PhET.from_lightning_checkpoint(ckpt_phet)
        self.h_dim = self.phet.config.h_dim
        self.resnet = ResNet18D1D.from_lightning_checkpoint(ckpt_resnet)
        
        # Freeze parameters
        if freeze_phet:
            self.phet.eval()
            for param in self.phet.parameters():
                param.requires_grad = False
        if freeze_resnet:
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Define projector
        self.n_embeds = n_embeds
        self.proj = nn.Sequential(
            nn.Linear(128, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_embeds*self.h_dim)
        )
        
        self.out_dir = out_dir
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, value_ids, phenotype_ids, flow_volume, mask, labels):
        # Encode spiro and project to embedding space
        spiro_embeds = self.resnet(flow_volume)
        spiro_embeds = self.proj(spiro_embeds)
        spiro_embeds = spiro_embeds.view(-1, self.n_embeds, self.h_dim)
        
        # Forward PhE-T
        logits = self.phet(value_ids=value_ids, phenotype_ids=phenotype_ids, embeds=spiro_embeds)['logits']
        logits = logits[:,self.n_embeds:,:]     # Ignore spiro logits
        logits = logits[mask]                   # Retrieve logits for the masked phenotype (Asthma)
        loss = F.cross_entropy(logits.view(-1, self.phet.config.v_size), labels.view(-1))
    
        # Compute risk score
        probs = F.softmax(logits, dim=-1)
        p_id = self.tokenizer.get_phenotype_id('Asthma')
        trait_info = self.tokenizer.boolean_traits[p_id]
        positive_probs = probs[:, trait_info['true_id']]
        negative_probs = probs[:, trait_info['false_id']]
        score = positive_probs / (positive_probs + negative_probs)
        
        return {'loss': loss, 'logits': logits, 'score': score}
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            value_ids=batch['value_ids'],
            phenotype_ids=batch['phenotype_ids'],
            flow_volume=batch['flow_volume'].unsqueeze(1),
            mask=batch['mask'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            value_ids=batch['value_ids'],
            phenotype_ids=batch['phenotype_ids'],
            flow_volume=batch['flow_volume'].unsqueeze(1),
            mask=batch['mask'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        y_hat = outputs['score']
        y = batch['label']
        self.validation_step_outputs.append({'loss': loss, 'y': y, 'y_hat': y_hat})
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        losses = torch.stack([x['loss'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        loss = torch.mean(losses)
        auroc = binary_auroc(y_hat, y.int())
        auprc = binary_auprc(y_hat, y.int())

        # Log results
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/auroc", auroc, sync_dist=True)
        self.log("val/auprc", auprc, sync_dist=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        outputs = self(
            value_ids=batch['value_ids'],
            phenotype_ids=batch['phenotype_ids'],
            flow_volume=batch['flow_volume'].unsqueeze(1),
            mask=batch['mask'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        y_hat = outputs['score']
        y = batch['label']
        eids = batch['eid']
        self.test_step_outputs.append({'loss': loss, 'y': y, 'y_hat': y_hat, 'eids': eids})

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        eids = torch.cat([x['eids'] for x in outputs])

        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, 'rs_asthma.json'), 'w') as f:
            f.write(json.dumps({
                "eids": eids.tolist(),
                "y_true": y.tolist(),
                "y_scores": y_hat.tolist()
            }))
        
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]