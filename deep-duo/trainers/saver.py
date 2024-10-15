import torch
import os

class Saver:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, model, optimizer, scheduler, filename):
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(filepath):
            print(f"Loading checkpoint '{filepath}'")
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{filepath}' (epoch {epoch})")
            return epoch
        else:
            print(f"No checkpoint found at '{filepath}'")
            return None
