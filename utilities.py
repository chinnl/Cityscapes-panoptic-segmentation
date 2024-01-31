from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, PolynomialLR, ReduceLROnPlateau
import os, re, glob
import matplotlib.pyplot as plt

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
    list_dir = glob.glob(save_dir + "*")
    if len(list_dir) > 0:
        list_increment = []
        for dir in list_dir:
            if re.search(r'\d+$', dir) is not None:
                list_increment.append((int(re.search(r'\d+$', dir).group()), re.search(r'\d+$', dir).span()[0]))
        if len(list_increment):
            max_increment = sorted(list_increment, key = lambda x: x[0], reverse = True)[0]  
                
        # if re.search(r'\d+$', save_dir) is not None:
        #     increment = int(re.search(r'\d+$', save_dir).group()) 
        #     loc = re.search(r'\d+$', save_dir).span()[0]
            os.makedirs(save_dir[:max_increment[1]] + "_" + str(max_increment[0] + 1))
            return save_dir[:max_increment[1]] + "_" + str(max_increment[0] + 1)
        else: 
            os.makedirs(save_dir + "_" + str(1))
            return save_dir + "_" + str(1)
    else:
        os.makedirs(save_dir)
        return save_dir
        
def write_log(save_dir, text):
    with open(os.path.join(save_dir, 'log.txt'), "a") as log:
        log.write(text)


def plot_results(training_result, fpath):
    if "Unnamed: 0" in training_result.columns:
        training_result = training_result.rename(columns={"Unnamed: 0":"iters"})
    fig, axes = plt.subplots(2,3, figsize = (20,10))
    for col_name, ax in zip(training_result.drop('iters', axis = 1), axes.flatten()):
        # sns.boxplot(training_result, x = 'iters', y = 'loss_sem_seg', ax = ax, orient = 'v')
        ax.plot(training_result['iters'], training_result[col_name])
        ax.set_title(col_name)
    fig.tight_layout()
    fig.savefig(fpath)