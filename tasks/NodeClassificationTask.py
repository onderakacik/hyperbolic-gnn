import torch as th
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
import numpy as np

from dataset.NodeClassificationDataset import NodeClassificationDataset
from models.NodeClassification import NodeClassification
from models.NodeClassificationExtended import NodeClassificationExtended
from utils import set_up_optimizer_scheduler
from dataset.AirportNodeClassificationDataset import AirportNodeClassificationDataset

def cross_entropy(log_prob, label, mask):
    label, mask = label.squeeze(), mask.squeeze()
    negative_log_prob = -th.sum(label * log_prob, dim=1)
    return th.sum(mask * negative_log_prob, dim=0) / th.sum(mask)

def get_f1_score(label, log_prob, mask):
    """Calculate F1 score for masked predictions"""
    label = label.squeeze()
    pred_class = th.argmax(log_prob, dim=1).cpu().numpy()
    real_class = th.argmax(label, dim=1).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    
    # Only calculate F1 score for masked (valid) nodes
    masked_pred = pred_class[mask == 1]
    masked_real = real_class[mask == 1]
    
    # Calculate macro F1 score
    f1 = f1_score(masked_real, masked_pred, average='macro')
    return f1

class NodeClassificationTask(pl.LightningModule):
    def __init__(self, args, rgnn, manifold):
        super(NodeClassificationTask, self).__init__()
        self.args = args
        self.manifold = manifold
        self.hyperbolic = False if args.select_manifold == "euclidean" else True
        self.rgnn = rgnn

        # Initialize dataset based on dataset type
        if args.task == 'airport':
            self.dataset = AirportNodeClassificationDataset(args)
        else:
            self.dataset = NodeClassificationDataset(args)
        
        # Choose the model based on args
        if getattr(args, "use_tangent", False):
            self.model = NodeClassificationExtended(args, rgnn, manifold)
        else:
            self.model = NodeClassification(args, rgnn, manifold)
        
        # Initialize metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Best metrics tracking
        self.best_val_f1 = 0.0
        self.test_f1_at_best_val = None
        
        # Disable automatic optimization for manual control
        self.automatic_optimization = False
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['rgnn', 'manifold'])

    def forward(self, adj, weight, features):
        return self.model(adj, weight, features)

    def training_step(self, batch, batch_idx):
        # Get optimizers
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        scores = self.forward(
            batch['adj'].long(),
            batch['weight'].float(),
            batch['features'].float()
        )
        
        loss = cross_entropy(
            scores,
            batch['y_train'].float(),
            batch['train_mask'].float()
        )
        
        # Manual optimization
        self.manual_backward(loss)
        
        if self.args.grad_clip > 0.0:
            th.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip)
        
        # Step optimizers
        optimizers[0].step()
        if self.hyperbolic and len(self.args.hyp_vars) != 0 and len(optimizers) > 1:
            optimizers[1].step()
            
        # Zero gradients
        for opt in optimizers:
            opt.zero_grad()

        # Calculate F1 score
        f1 = get_f1_score(
            batch['y_train'].float(),
            scores,
            batch['train_mask'].float()
        )
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.training_step_outputs.append({'loss': loss, 'f1': f1})
        
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self.forward(
            batch['adj'].long(),
            batch['weight'].float(),
            batch['features'].float()
        )
        
        val_f1 = get_f1_score(
            batch['y_val'].float(),
            scores,
            batch['val_mask'].float()
        )
        
        self.log('val_f1', val_f1, prog_bar=True)
        self.validation_step_outputs.append(val_f1)
        return val_f1

    def test_step(self, batch, batch_idx):
        scores = self.forward(
            batch['adj'].long(),
            batch['weight'].float(),
            batch['features'].float()
        )
        
        test_f1 = get_f1_score(
            batch['y_test'].float(),
            scores,
            batch['test_mask'].float()
        )
        
        self.log('test_f1', test_f1, prog_bar=True)
        self.test_step_outputs.append(test_f1)
        return test_f1

    def configure_optimizers(self):
        optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = set_up_optimizer_scheduler(
            self.hyperbolic, self.args, self.model
        )
        
        if self.hyperbolic and len(self.args.hyp_vars) != 0:
            return [optimizer, hyperbolic_optimizer]
        return optimizer

    def train_dataloader(self):
        # dataset = NodeClassificationDataset(self.args)
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()

    def run(self):
        # Initialize WandbLogger
        wandb_logger = WandbLogger(
            project="hyperbolic-gnn-airport",
            name=f"node_classification_{self.args.select_manifold}_{self.args.use_tangent}",
            config=vars(self.args)
        )
        
        # Configure checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_f1',
            dirpath='checkpoints',
            filename='best_node_classification_model',
            save_top_k=1,
            mode='max',
        )
        
        # Add early stopping callback
        early_stopping = pl.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=self.args.patience,  # Using patience from args
            mode='max',
            verbose=True
        )
        
        # Initialize trainer with early stopping
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            devices=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback, early_stopping],  # Added early stopping callback
            logger=wandb_logger,
        )
        
        # Train model
        trainer.fit(self)
        
        # Print final results
        print(f"\nTraining completed!")
        print(f"Best validation F1 score: {checkpoint_callback.best_model_score:.4f}")
        
        # Load and test best model
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from {best_model_path}")
        best_model = NodeClassificationTask.load_from_checkpoint(
            best_model_path,
            args=self.args,
            rgnn=self.rgnn,
            manifold=self.manifold
        )
        
        # Test best model
        print("\nEvaluating best model on test set...")
        trainer.test(best_model)
