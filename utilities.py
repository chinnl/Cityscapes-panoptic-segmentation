from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, PolynomialLR, ReduceLROnPlateau
import os
import re


def get_optimizer_by_name(cfg, model):
    if cfg.name == 'Adam':
        return Adam(params = model.parameters(),
                    lr = cfg.base_lr,
                    )
    elif cfg.name == 'AdamW':
        return AdamW(params = model.parameters(),
                     lr = cfg.base_lr,
                     )
    elif cfg.name == 'SGD':
        return SGD(params = model.parameters(),
                   lr = cfg.base_lr,
                   )
    else:
        raise ValueError("Invalid optimizer, expected Adam|AdamW|SGD but got {}".format(cfg.name))

def get_scheduler_by_name(cfg, optimizer, max_iters):
    if cfg.name == 'StepLR':
        return StepLR(optimizer=optimizer,
                      step_size=cfg.step,
                      gamma=cfg.lr_factor,
                      verbose=False,)
    elif cfg.name == 'PolynomialLR':
        return PolynomialLR(optimizer=optimizer,
                            total_iters=max_iters,
                            power=1.0,
                            verbose=False)
    elif cfg.name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer=optimizer,
                                 mode = min,
                                 factor = cfg.lr_factor,
                                 verbose=False)
    elif cfg.name is None:
        return None
    
    else:
        raise ValueError("Invalid scheduler, expected StepLR|PolynomialLR|ReduceLROnPlateau|None but got {}".format(cfg.name))

def create_dir(save_dir):
    if os.path.isdir(save_dir):
        if re.search(r'\d+$', save_dir) is not None:
            increment = int(re.search(r'\d+$', save_dir).group()) 
            loc = re.search(r'\d+$', save_dir).span()[0]
            os.makedirs(save_dir[:loc] + "_" + str(increment + 1))
            return save_dir[:loc] + "_" + str(increment + 1)
        else: 
            os.makedirs(save_dir + "_" + str(1))
            return save_dir + "_" + str(1)
    else:
        os.makedirs(save_dir)
        return save_dir
        