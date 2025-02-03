import torch as th
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataloader import default_collate
from pytorch_lightning.loggers import WandbLogger

from models.GraphPrediction import GraphPrediction
from models.GraphPredictionExtended import GraphPredictionExtended
from dataset.GraphDataset import GraphDataset
from dataset.SyntheticDataset import SyntheticDataset
from utils import pad_sequence, set_up_optimizer_scheduler  # Make sure these exist

def collate_fn(batch):
    max_neighbor_num = -1
    for data in batch:
        for row in data['adj_mat']:
            max_neighbor_num = max(max_neighbor_num, len(row))
    for data in batch:
        # Pad the adjacency list and corresponding weights
        data['adj_mat'] = pad_sequence(data['adj_mat'], maxlen=max_neighbor_num)
        data['weight'] = pad_sequence(data['weight'], maxlen=max_neighbor_num)
        data['node'] = np.array(data['node']).astype(np.float32)
        data['adj_mat'] = np.array(data['adj_mat']).astype(np.int32)
        data['weight'] = np.array(data['weight']).astype(np.float32)
        data['label'] = np.array(data['label'])
    return default_collate(batch)

class GraphPredictionTask(pl.LightningModule):
    def __init__(self, args, rgnn, manifold):
        """
        Implements a Graph Prediction task as a LightningModule.
        The underlying model is an instance of GraphPrediction.
        """
        super(GraphPredictionTask, self).__init__()
        self.args = args
        self.rgnn = rgnn
        self.manifold = manifold
        self.hyperbolic = (args.select_manifold != "euclidean")
        
        # Initialize lists for collecting metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Choose the loss function based on the task type.
        if args.is_regression:
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()
        
        # Choose the model based on args
        if getattr(args, "use_tangent", False):
            self.model = GraphPredictionExtended(args, rgnn, manifold)
        else:
            self.model = GraphPrediction(args, rgnn, manifold)
        
        # When using multiple optimizers (to step hyperbolic variables separately)
        # disable Lightning's automatic optimization.
        self.automatic_optimization = False
        
        # Add tracking of best validation loss and corresponding test loss
        self.best_val_loss = float('inf')
        self.test_loss_at_best_val = None
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['rgnn', 'manifold'])

    def load_dataset(self, dataset_class):
        train_dataset = dataset_class(self.args, split='train')
        dev_dataset = dataset_class(self.args, split='dev')
        test_dataset = dataset_class(self.args, split='test')
        
        # If in debug mode, sample only a fraction of data.
        if self.args.debug:
            train_dataset.dataset = train_dataset.dataset[:int(len(train_dataset.dataset) * 0.1)]
            dev_dataset.dataset = dev_dataset.dataset[:int(len(dev_dataset.dataset) * 0.1)]
            test_dataset.dataset = test_dataset.dataset[:int(len(test_dataset.dataset) * 0.1)]
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                  collate_fn=collate_fn, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=self.args.batch_size,
                                collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size,
                                 collate_fn=collate_fn, num_workers=0)
        return train_loader, dev_loader, test_loader

    def load_data(self):
        if self.args.task == 'synthetic':
            return self.load_dataset(SyntheticDataset)
        else:
            return self.load_dataset(GraphDataset)
    
    def forward(self, node, adj_mat, weight, mask):
        """
        Forward pass simply calls the underlying GraphPrediction model.
        """
        return self.model(node, adj_mat, weight, mask)
    
    def training_step(self, batch, batch_idx):
        if "mask" in batch:
            mask = batch["mask"].int()
        else:
            mask = th.tensor([batch["adj_mat"].shape[1]], device=self.device).int()
        
        scores = self.forward(batch["node"].float(),
                            batch["adj_mat"].long(),
                            batch["weight"].float(),
                            mask)
        
        # Handle property indexing for regression and classification
        if self.args.is_regression:
            # Scale the scores using mean and std if provided
            scaled_scores = scores.view(-1)
            if hasattr(self.args, 'std') and hasattr(self.args, 'mean'):
                scaled_scores = scaled_scores * self.args.std[self.args.prop_idx] + self.args.mean[self.args.prop_idx]
            
            target = batch["label"].view(-1)[self.args.prop_idx].float().to(self.device)
            loss = self.loss_function(scaled_scores, target)
        else:
            target = batch["label"].view(-1)[self.args.prop_idx].long().to(self.device)
            loss = self.loss_function(scores, target)
        
        logged_loss = loss
        if self.args.is_regression:
            if self.args.metric == "mae":
                logged_loss = th.sqrt(loss)

        # MANUAL OPTIMIZATION:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        
        # backward pass
        self.manual_backward(loss)
        
        # Grad clipping if specified.
        if self.args.grad_clip > 0.0:
            th.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        
        # Step the primary optimizer.
        optimizers[0].step()
        
        # If hyperbolic optimization is necessary, step the second optimizer.
        if (self.hyperbolic and hasattr(self.args, "hyp_vars") and
            len(self.args.hyp_vars) != 0 and len(optimizers) > 1):
            optimizers[1].step()
        
        # Zero the gradients.
        for opt in optimizers:
            opt.zero_grad()

        # self.log("train_loss", logged_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.training_step_outputs.append(logged_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if "mask" in batch:
            mask = batch["mask"].int()
        else:
            mask = th.tensor([batch["adj_mat"].shape[1]], device=self.device).int()
        
        scores = self.forward(batch["node"].float(),
                            batch["adj_mat"].long(),
                            batch["weight"].float(),
                            mask)
        
        if self.args.is_regression:
            scaled_scores = scores.view(-1)
            if hasattr(self.args, 'std') and hasattr(self.args, 'mean'):
                scaled_scores = scaled_scores * self.args.std[self.args.prop_idx] + self.args.mean[self.args.prop_idx]
            
            target = batch["label"].view(-1)[self.args.prop_idx].float().to(self.device)
            loss = self.loss_function(scaled_scores, target)
        else:
            target = batch["label"].view(-1)[self.args.prop_idx].long().to(self.device)
            loss = self.loss_function(scores, target)
        
        logged_loss = loss
        if self.args.is_regression:
            if self.args.metric == "mae":
                logged_loss = th.sqrt(loss)

        self.validation_step_outputs.append(logged_loss)
        return loss

    def test_step(self, batch, batch_idx):
        if "mask" in batch:
            mask = batch["mask"].int()
        else:
            mask = th.tensor([batch["adj_mat"].shape[1]], device=self.device).int()

        scores = self.forward(batch["node"].float(),
                            batch["adj_mat"].long(),
                            batch["weight"].float(),
                            mask)
        
        if self.args.is_regression:
            scaled_scores = scores.view(-1)
            if hasattr(self.args, 'std') and hasattr(self.args, 'mean'):
                scaled_scores = scaled_scores * self.args.std[self.args.prop_idx] + self.args.mean[self.args.prop_idx]
            
            target = batch["label"].view(-1)[self.args.prop_idx].float().to(self.device)
            loss = self.loss_function(scaled_scores, target)
        else:
            target = batch["label"].view(-1)[self.args.prop_idx].long().to(self.device)
            loss = self.loss_function(scores, target)
        
        logged_loss = loss
        if self.args.is_regression:
            if self.args.metric == "mae":
                logged_loss = th.sqrt(loss)

        self.test_step_outputs.append(logged_loss)
        return logged_loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            epoch_val_loss = th.stack(self.validation_step_outputs).mean()
            if self.args.is_regression and self.args.metric == "rmse":
                epoch_val_loss = th.sqrt(epoch_val_loss)
            if epoch_val_loss.item() < self.best_val_loss:
                self.best_val_loss = epoch_val_loss.item()
            self.log("val_loss", epoch_val_loss, prog_bar=True)
            if self.logger:
                self.logger.log_metrics({
                    "val_loss": epoch_val_loss.item()
                }, step=self.trainer.current_epoch)
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) > 0:
            epoch_test_loss = th.stack(self.test_step_outputs).mean()
            # Apply RMSE conversion at epoch end if needed
            if self.args.is_regression and self.args.metric == "rmse":
                epoch_test_loss = th.sqrt(epoch_test_loss)
            self.log("test_loss", epoch_test_loss, prog_bar=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Returns both optimizers and their schedulers.
        If hyperbolic training is on (and hyp_vars are provided) then we return two optimizers.
        """
        optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = set_up_optimizer_scheduler(
            self.hyperbolic, self.args, self.model
        )
        
        if self.hyperbolic and hasattr(self.args, "hyp_vars") and len(self.args.hyp_vars) != 0:
            # Ensure both optimizers are not None
            if optimizer is None or hyperbolic_optimizer is None:
                raise ValueError("Optimizers must not be None.")
            
            # For multiple optimizers, just return the optimizers if schedulers aren't properly configured
            if not hasattr(lr_scheduler, 'optimizer') or not hasattr(hyperbolic_lr_scheduler, 'optimizer'):
                return [optimizer, hyperbolic_optimizer]
            
            return [optimizer, hyperbolic_optimizer]
        else:
            # Ensure optimizer is not None
            if optimizer is None:
                raise ValueError("Optimizer must not be None.")
            
            # For single optimizer, just return the optimizer if scheduler isn't properly configured
            if lr_scheduler is None or not hasattr(lr_scheduler, 'optimizer'):
                return optimizer
            
            return optimizer

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) > 0:
            epoch_train_loss = th.stack(self.training_step_outputs).mean()
            if self.args.is_regression and self.args.metric == "rmse":
                epoch_train_loss = th.sqrt(epoch_train_loss)
            if self.logger:
                self.logger.log_metrics({
                    "train_loss": epoch_train_loss.item()
                }, step=self.trainer.current_epoch)
        self.training_step_outputs.clear()

    def run(self):
        train_loader, val_loader, test_loader = self.load_data()
        
        # Initialize WandbLogger
        wandb_logger = WandbLogger(
            project="hyperbolic-gnn",  # Change this to your project name
            name=f"{self.args.task}_{self.args.select_manifold}_{self.args.prop_idx}_{self.args.use_tangent}_report",  # Run name
            config=vars(self.args)  # Log all arguments
        )
        
        # Configure checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='best_model',
            save_top_k=1,
            mode='min',
        )
        
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            devices=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
            logger=wandb_logger,  # Add the wandb logger
        )
        
        # Train the model
        trainer.fit(self, train_loader, val_loader)

        # Print final results
        print(f"\nTraining completed!")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")
        
        # Load the best model - FIXED THIS PART
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from {best_model_path}")
        best_model = GraphPredictionTask.load_from_checkpoint(
            best_model_path,
            args=self.args,
            rgnn=self.rgnn,
            manifold=self.manifold
        )
        
        # Run test with best model and print results
        print("\nEvaluating best model on test set...")
        trainer.test(best_model, test_loader)
        