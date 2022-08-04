from collections import defaultdict
from tracemalloc import start
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model.model import *


def train_model(model, optimizer, 
                train_loader, test_loader,
                num_epochs, writer=None, 
                starting_epoch=0, 
                starting_batch_train=0, 
                mean = None, std = None,
                gradient_type='normal', gradient_clip=1.0,
                computeEndReg=False,
                save_folder=None, save_every = 20, 
                save_type="inference",
                device = 'cuda'):
    
    model.to(device)
    train_losses = {}
    eval_losses = {}
    batch_ct_train = starting_batch_train

    print(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in tqdm(range(starting_epoch, num_epochs)):
        
        # ---TRAINING---
        model.train()
        for _, inputs in enumerate(train_loader):
            
            inputs = inputs[0].to(device)
            optimizer.zero_grad()
            _, _, loss = model.loss(inputs, computeEndReg=computeEndReg)
            loss.backward()
            
            # Clip gradient depending on type
            if gradient_clip:
                if gradient_type == "normal":
                    nn.utils.clip_grad_norm_(model.parameters(), 
                                            max_norm=gradient_clip, norm_type=2)
                elif gradient_type == "value":
                    nn.utils.clip_grad_value_(model.parameters(), 
                                            clip_value=gradient_clip)
            
            optimizer.step()
            
            # Save loss
            for k, v in model.result_dict.items():
                train_losses.setdefault(k, []).append(v)
                if writer: # write to tensorboard 
                    writer.add_scalar(f"Loss/train/{k}", v, batch_ct_train)
            batch_ct_train += 1
                
        if save_folder:
            if (epoch+1) % save_every == 0 or epoch+1 == num_epochs:
                if save_type == "inference":
                    torch.save(model.state_dict(), f"{save_folder}/model_{save_type}_E{epoch+1}")
                elif save_type == "checkpoint":
                    torch.save({
                        'epoch': epoch+1,
                        'batch_count_train' : batch_ct_train,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'mean' : mean,
                        'std' : std
                    }, f"{save_folder}/model_{save_type}_E{epoch+1}")
             
        # ---VALIDATION---
        model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                _, _, loss = model.loss(inputs, computeEndReg = computeEndReg)
                
                # Save loss
                for k, v in model.result_dict.items():
                    eval_losses.setdefault(k, []).append(v)
                    if writer: # write to tensorboard
                        writer.add_scalar(f"Loss/eval/{k}", v, epoch+1)
            
    return train_losses, eval_losses
