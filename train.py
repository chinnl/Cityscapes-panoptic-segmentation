from model_from_config import Panoptic_FPN_R50_Mask_RCNN
from Data.data_loader import build_dataloader
import anyconfig, munch
import os
from utilities import get_optimizer_by_name, get_scheduler_by_name, create_dir
import torch
import pandas as pd
from tqdm import tqdm

config_path = 'config.yaml'
config = anyconfig.load(config_path)
config = munch.munchify(config)

max_iters = config.config.max_iters
model = Panoptic_FPN_R50_Mask_RCNN(config.model)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

optim = get_optimizer_by_name(config.config.optimizer, model)
scheduler = get_scheduler_by_name(config.config.scheduler, optimizer=optim, max_iters=max_iters)
trainloader, valloader = build_dataloader(config.data)
save_dir = create_dir(config.config.save_dir)

train_losses = {'loss_sem_seg': [],
                'loss_rpn_cls': [],
                'loss_rpn_loc': [],
                'loss_cls': [],
                'loss_box_reg': [],
                'loss_mask': []}
val_losses = {'loss_sem_seg': [],
                'loss_rpn_cls': [],
                'loss_rpn_loc': [],
                'loss_cls': [],
                'loss_box_reg': [],
                'loss_mask': []}
best_val_loss = 100

with open(os.path.join(save_dir, 'log.txt'), "w") as log:
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
        
        for batched_input in trainloader:
            batch_losses = model(batched_input)
            total_train_loss = torch.tensor(0).to(model.device)
            for _, v in batch_losses.items():
                total_train_loss = total_train_loss + v
                
            optim.zero_grad()
            total_train_loss.backward()
            optim.step()
            scheduler.step()
            
            for key in epoch_train_losses.keys():
                epoch_train_losses[key] += batch_losses[key].item()
        
        train_log = " - ".join(["{}: {:.3f}".format(k, v/len(trainloader)) for k, v in epoch_train_losses.items()])
        print(f"Epoch {epoch}/{max_iters}: \n Train: {train_log}")
                
        with torch.no_grad():
            for batched_input in tqdm(valloader):
                batch_losses = model(batched_input)
                
                for key in epoch_val_losses.keys():
                    epoch_val_losses[key] += batch_losses[key].item()
                
        for key in train_losses.keys():
            train_losses[key].append(epoch_train_losses[key]/len(trainloader))
        
        for key in val_losses.keys():
            val_losses[key].append(epoch_val_losses[key]/len(valloader))
        
        val_log = " - ".join(["{}: {:.3f}".format(k, v/len(valloader)) for k, v in epoch_val_losses.items()])
        print(f"Val: {val_log} \n" + "-"*50)
        
        if epoch%config.config.save_period == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"sd_epoch_{epoch}.pt"))
        
        last_total_val_loss = 0
        for _, v in epoch_val_losses.items():
            last_total_val_loss = last_total_val_loss + v/len(valloader)
        
        if last_total_val_loss <= best_val_loss:
            print(f"Save best checkpoint at epoch {epoch}, total val loss: {last_total_val_loss}")
            torch.save(model.state_dict(), os.path.join(save_dir, f"best.pt"))
            best_val_loss = last_total_val_loss
            
        log.write(f"Epoch {epoch}/{max_iters}: \n Train: {train_log} \n Val: {val_log}" + "-"*50)


train_log_pd = pd.DataFrame.from_dict(train_losses)
train_log_pd.to_csv(os.path.join(save_dir, 'train_losses.txt'))

val_log_pd = pd.DataFrame.from_dict(val_losses)
val_log_pd.to_csv(os.path.join(save_dir, 'val_losses.txt'))