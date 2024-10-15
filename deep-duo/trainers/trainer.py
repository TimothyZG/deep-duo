import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import copy

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, logger, saver, num_epochs, dataset_name):
        self.model = model
        self.dataloaders = dataloaders  # Should include 'train' and 'val' (and optionally 'test')
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.saver = saver
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name.lower()
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_metric = 0.0  # Best F1 or accuracy, depending on dataset

    def train(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)

            # Training phase
            train_loss, train_metric = self._train_epoch(epoch)
            print(f'Train Loss: {train_loss:.4f} Metric: {train_metric:.4f}')

            # Validation phase
            val_loss, val_metric = self._validate_epoch(epoch)
            print(f'Val Loss: {val_loss:.4f} Metric: {val_metric:.4f}')

            # Scheduler step
            self.scheduler.step()

            # Check if this is the best model so far
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                # Save the best model using saver
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_metric': self.best_metric,
                }
                self.saver.save_checkpoint(state, filename=f'best_model_{self.dataset_name}.pth')

            # Log metrics using logger
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            if self.dataset_name == 'iwildcam':
                metrics['train_f1'] = train_metric
                metrics['val_f1'] = val_metric
            else:  # Assuming 'rxrx1' or others
                metrics['train_accuracy'] = train_metric
                metrics['val_accuracy'] = val_metric

            self.logger.log(metrics)

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.best_metric

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(tqdm(self.dataloaders['train'])):
            data, targets, metadata = batch
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # Log every 100 batches
            if batch_idx % 100 == 0:
                batch_loss = running_loss / ((batch_idx + 1) * data.size(0))
                batch_metric = self._compute_metric(all_labels, all_preds)
                self.logger.log({
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'loss': batch_loss,
                    'metric': batch_metric,
                })

        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epoch_metric = self._compute_metric(all_labels, all_preds)

        return epoch_loss, epoch_metric

    def _validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val']):
                data, targets, metadata = batch
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * data.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epoch_metric = self._compute_metric(all_labels, all_preds)

        return epoch_loss, epoch_metric

    def _compute_metric(self, labels, preds):
        if self.dataset_name == 'iwildcam':
            # Compute F1 score
            return f1_score(labels, preds, average='macro', zero_division=0)
        else:  # Assuming 'rxrx1' or others
            # Compute accuracy
            return accuracy_score(labels, preds)
