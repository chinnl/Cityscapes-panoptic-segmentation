from model_from_config import load_model
from Data.data_loader import build_dataloader
import anyconfig, munch
import os
from utilities import get_optimizer_by_name, create_dir, write_log, plot_results
import torch
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math

config_path = 'config.yaml'
config = anyconfig.load(config_path)
config = munch.munchify(config)   

max_iters = config.config.max_iters
model = load_model(config.model)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
print("Load model: Done")

base_lr = config.config.optimizer.base_lr
warmup_steps = config.config.warmup_steps

def warmup_scheduler(current_step: int):
    if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
        return float(current_step/ warmup_steps)
    else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return base_lr*max_iters*(1+math.cos(math.pi*current_step/max_iters))/2
    
optim = get_optimizer_by_name(config.config.optimizer, model)
# scheduler = get_scheduler_by_name(config.config.scheduler, optimizer=optim, max_iters=max_iters)
scheduler = LambdaLR(optim, lr_lambda=warmup_scheduler)
trainloader, valloader = build_dataloader(config.data)
print("Load data: Done")

save_dir = create_dir(config.config.save_dir)
train_losses = {'iters': [],
                'loss_sem_seg': [],
                'loss_rpn_cls': [],
                'loss_rpn_loc': [],
                'loss_cls': [],
                'loss_box_reg': [],
                'loss_mask': []}
val_losses = {  'iters': [],
                'loss_sem_seg': [],
                'loss_rpn_cls': [],
                'loss_rpn_loc': [],
                'loss_cls': [],
                'loss_box_reg': [],
                'loss_mask': []}
best_val_loss = 100
# batched_input = next(iter(trainloader))
   
for epoch in range(1, max_iters+1):
    model.train()
    epoch_train_losses = {'loss_sem_seg': 0,
                            'loss_rpn_cls': 0,
                            'loss_rpn_loc': 0,
                            'loss_cls': 0,
                            'loss_box_reg': 0,
                            'loss_mask': 0}
    epoch_val_losses = {'loss_sem_seg': 0,
                            'loss_rpn_cls': 0,
                            'loss_rpn_loc': 0,
                            'loss_cls': 0,
                            'loss_box_reg': 0,
                            'loss_mask': 0}
    
    for batched_input in tqdm(trainloader):
        batch_losses = model(batched_input)
        total_train_loss = torch.tensor(0).to(model.device)
        for k, v in batch_losses.items():
            total_train_loss = total_train_loss + v
            
        optim.zero_grad()
        total_train_loss.backward()
        optim.step()
        scheduler.step()
        
        for key in epoch_train_losses.keys():
            epoch_train_losses[key] += batch_losses[key].item()
    
    train_log = " - ".join(["{}: {:.4f}".format(k, v/len(trainloader)) for k, v in epoch_train_losses.items()]) #+ f" - Total loss: {total_train_loss.item()}"
    print(f"Epoch {epoch}/{max_iters}: \n Train: {train_log}")
    
    for key in train_losses.keys():
        if key == 'iters':
            train_losses[key].append(epoch)
            continue
        train_losses[key].append(epoch_train_losses[key]/len(trainloader))
              
    with torch.no_grad():
        for batched_input in tqdm(valloader):
            batch_losses = model(batched_input)
            
            for key in epoch_val_losses.keys():
                epoch_val_losses[key] += batch_losses[key].item()
    
    for key in val_losses.keys():
        if key == 'iters':
            val_losses[key].append(epoch)
            continue
        val_losses[key].append(epoch_val_losses[key]/len(valloader))
    
    val_log = " - ".join(["{}: {:.4f}".format(k, v/len(valloader)) for k, v in epoch_val_losses.items()])
    print(f"Val: {val_log} \n" + "-"*50)
    
    if epoch%config.config.save_period == 0 and model.gpu_id == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"sd_epoch_{epoch}.pt"))
    
    last_total_val_loss = 0
    for _, v in epoch_val_losses.items():
        last_total_val_loss = last_total_val_loss + v/len(valloader)
    
    if last_total_val_loss <= best_val_loss:
        print(f"Save best checkpoint at epoch {epoch}, total val loss: {last_total_val_loss}")
        write_log(save_dir, "Save best checkpoint at epoch {}, total val loss: {:.4f}".format(epoch, last_total_val_loss))
        if model.gpu_id == 0: torch.save(model.state_dict(), os.path.join(save_dir, f"best.pt"))
        best_val_loss = last_total_val_loss
        
    write_log(save_dir, f"Epoch {epoch}/{max_iters}: \n Train: {train_log} \n Val: {val_log} \n")


train_log_pd = pd.DataFrame.from_dict(train_losses)
train_log_pd.to_csv(os.path.join(save_dir, 'train_losses.csv'))

val_log_pd = pd.DataFrame.from_dict(val_losses)
val_log_pd.to_csv(os.path.join(save_dir, 'val_losses.csv'))
plot_results(train_log_pd, save_dir)