import numpy as np
import torch, json, os 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, name, patience=10, verbose=False, delta=0):
        """
        Args:
            name (str): Name of the saved model.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.name =  name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, epoch, val_loss, model, parameters, grid_result):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, parameters, grid_result)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, parameters, grid_result)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, parameters, grid_result=False):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if grid_result:
            print('.............. Saving Parameters..............')
            json.dump(parameters,  open('final_final_results/'+ self.name + '.json','w'))
        
        torch.save(model, 'final_final_results/final_models/'+ self.name + "_model_epoch_"+ str(epoch) + ".pt")
        # if epoch >1:
        #      os.remove('final_final_results/final_models/'+ self.name + "_model_epoch_"+ str(epoch-1) + ".pt")
        self.val_loss_min = val_loss