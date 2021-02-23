import torch
import numpy as np
import pandas as pd

import logging    
from logger import get_logger
from tqdm import tqdm



import json, os, sys
import time

sys.path.append('/')
from utils.utils import fetch_dataloader , lag_data
from utils.metrics import iou_pytorch, pixel_segementation_evaluation , confusion_matrix
from utils.PytorchEarlyStopping import EarlyStopping
from utils.loss import weighted_binary_cross_entropy

from model.model  import  ImageLSTM, ImageGRU, ImageRNN, CNN, UNet, ESN

# for reproducibility
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

# dataset used
dataset= 'CIFAR_100' # 'CIFAR_10' , 'BSR', 'WEIZMANN'

# name of the model
model= 'RNN' # 'RNN', 'LSTM', 'GRU', 'ESN'

# Model instance
batch_size      = 100
image_dimension = 32
epochs          = 10000
lag             = 1
load_all_original_images= '0'

# path to orignal images
original_image_directory     = f'../../Data/FINAL_DATA/HETEROGENEOUS/{dataset}/'
# path to segmentation images
segmentation_image_directory = f'../../Data/FINAL_DATA/HETEROGENEOUS/{dataset}/SEGMENTATION_DATA/BINARY_SEGMENTATION/'

# device to perform computation (CPU or GPU)
device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parameters to the dataloader
parameters = {
"original_image_path":original_image_directory,
"segmentation_image_path":segmentation_image_directory,
"image_dimension":image_dimension,
"batch_size":batch_size,
"total_segmentation_images":30,
"total_original_images": 1 if load_all_original_images=='0' else None, # load all the images if NONE given
'threshold':0.5,
'window':1,
'extentions':['.png', '.jpg'] if dataset=='WEIZMANN' else ['.jpg', '.jpg'],
"shuffle":False,
"pin_memory":False,
"num_workers": 0 if sys.platform =='win32' else 4
}

# use GPU if available
parameters["cuda"] = torch.cuda.is_available()
dataloaders        = fetch_dataloader(parameters)




filename = f'{model}_{dataset}_grid'

# gridsearch space
gridsearch_space = {
            'learning_rate': [0.01, 0.001, 0.1],
            'hidden_size':  [256, 512, 1024],
                    }


def model_hypersearch_fit( gridsearch_space, 
                    train_data, 
                    test_data,
                    loss_function,
                    image_dimension,
                    epochs,
                    filename):
    # number of inputs nodes to the reservoir
    n_inputs          = image_dimension
    # number of output nodes from the reservoir
    n_outputs         = image_dimension*image_dimension
    n_layers          = 1
    criterion         = loss_function 
    epochs            = epochs
    hidden_size       = gridsearch_space['hidden_size']
    learning_rate     = gridsearch_space['learning_rate']
    max_iou           = -np.inf
     
    # capture the time to start training
    epoch_start_time = time.time()
    # to track the training loss as the model trains
    train_running_loss     = []
    # to track the validation loss as the model trains
    valid_running_loss     = []
    # to track the average training loss per epoch as the model trains
    train_epoch_loss       = []
    # to track the average validation loss per epoch as the model trains
    valid_epoch_loss       = [] 
    # to track the validation dice as the model trains
    valid_running_dice     = []
    # to track the validation accuracy as the model trains
    valid_running_accuracy = []
    # to track the validation precision as the model trains
    valid_running_precision = []
    # to track the validation recall as the model trains
    valid_running_recall   = []
    # to track the average validation dice per epoch as the model trains
    valid_epoch_dice       = []
    # to track the average validation accuracy per epoch as the model trains
    valid_epoch_accuracy   = []
    # to track the average validation precision per epoch as the model trains
    valid_epoch_precision   = []
    # to track the average validation recall per epoch as the model trains
    valid_epoch_recall      = []
     # epochs
    epoch_list              = []

    try:
        for lr_ixd, lr in enumerate(learning_rate):   
            for hs_ixd, hs in enumerate(hidden_size):
                    early_stopping = EarlyStopping(name=str(filename),patience=10, verbose=True)    

                    # save grid last step for next reload (incase of interuption)
                    save_grid_parameters(gridsearch_space, lr_ixd, hs_ixd, filename)
                    # re-initialize model with new parameters
                    model       = ImageRNN(n_inputs*2, 
                                            n_outputs, n_hidden=int(hs),
                                            n_layers=n_layers, 
                                            bidirectional=False).to(device)
                    # initliaze optimizer along with data loaders
                    optimizer=torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
                
                    for epoch in tqdm(np.arange(1, epochs+1)):
                        model.train()
                        for batch_idx, (inputs, labels, names, ratio) in enumerate(train_data):
                            
                            # load data and move data to GPU's
                            inputs = inputs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                            # lag the data n steps ahead
                            inputs,labels = lag_data(inputs,labels, lag)

                            inputs, labels = reshape_data(inputs, labels)
                
                            # initialize the hidden state on the GPU's
                            model.init_hidden(inputs, device)
                            # forward-propogation
                            outputs = model(inputs)
                            # outputs = outputs.view(-1, 1,image_dimension*image_dimension)
                            # outputs = outputs[:,1]
            
                            # print(labels.shape, outputs.shape)
                            loss = criterion(labels.view(-1, image_dimension,image_dimension), 
                            outputs.view(-1, image_dimension,image_dimension))

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                            optimizer.step()
                            optimizer.zero_grad()
                            train_running_loss.append(loss.detach().item())
                            
                            # clear variables from memory
                            del inputs, labels, outputs 
                            torch.cuda.empty_cache()

                        # test on validation-testing data                    
                        model.eval()
                        with torch.no_grad(): # do not calculate gradient and save memory usage
                            for batch_idx, (inputs, labels, names, ratio) in enumerate(test_data):

                                # load data and move data to GPU's
                                inputs = inputs.to(device, non_blocking=True)
                                labels = labels.to(device, non_blocking=True)

                                # shift the data to form a sequence. (inputs(t), labels(t+1))
                                inputs, labels = reshape_data(inputs, labels)

                                # forward-propogation
                                model.init_hidden(inputs, device)

                                # forward-propogation
                                outputs = model(inputs) 
                                outputs = outputs.view(-1, 1, image_dimension*image_dimension)
                                # outputs = outputs[:,1]

                                val_loss    = criterion(labels.view(-1, image_dimension*image_dimension), 
                                                    outputs.view(-1, image_dimension*image_dimension)).detach().item()

                                outputs = (outputs >= 0.5)*1
                                iou     =  iou_pytorch(outputs.view(-1, image_dimension, image_dimension),  
                                                        labels.view(-1, image_dimension, image_dimension)).detach().item()

                                f1, precision, recall  = pixel_segementation_evaluation(labels.cpu().detach().numpy().reshape(-1),
                                                    outputs.cpu().detach().numpy().reshape(-1))

                                # print training/validation statistics
                                valid_running_loss.append(val_loss)
                                valid_running_accuracy.append(f1)
                                valid_running_precision.append(precision)
                                valid_running_recall.append(recall)
                                valid_running_dice.append(iou)
                                
                                # clear variables from memory
                                del inputs, labels, outputs 
                                torch.cuda.empty_cache()

                            # print training/validation statistics 
                            # calculate average loss over an epoch
                            train_loss      = np.mean(train_running_loss)
                            valid_loss      = np.mean(valid_running_loss)
                            valid_dice      = np.mean(valid_running_dice)
                            valid_accuracy  = np.mean(valid_running_accuracy)
                            valid_precision = np.mean(valid_running_precision)
                            valid_recall    = np.mean(valid_running_recall)

                            train_epoch_loss.append(train_loss)
                            valid_epoch_loss.append(valid_loss)
                            valid_epoch_dice.append(valid_dice)
                            valid_epoch_accuracy.append(valid_accuracy)
                            valid_epoch_precision.append(valid_precision)
                            valid_epoch_recall.append(valid_recall)
                            epoch_list.append(epoch)

                            msg ='Epoch: {}, Time_taken: {:.3f} minutes, Training Loss: {:.3f}, Validation Loss: {:.3f}, \
                            Validation precision: {:.3f}, Validation recall: {:.3f}, Validation f1 score: {:.3f}, Validation IoU: {:.3f}'.format(                
                            epoch, (time.time()-epoch_start_time)/60, train_loss, valid_loss, 
                            valid_precision, valid_recall, valid_accuracy, valid_dice)

                            

                            # clear lists to track next epoch
                            train_running_loss      = []
                            valid_running_loss      = []
                            valid_running_dice      = []
                            valid_running_accuracy  = []
                            valid_running_precision = []
                            valid_running_recall    = []
                            
                            # early_stopping needs the validation loss to check if it has decresed, 
                            # and if it has, it will make a checkpoint of the current model
                            if( valid_dice > max_iou) and (early_stopping.early_stop is False):
                                best_grid_parameters = dict()
                                best_grid_parameters['learning_rate']=float(lr)
                                best_grid_parameters['hidden_size']=float(hs)
                                best_grid_parameters['iou']=float(iou)
                                best_grid_parameters['f1_score']=float(f1)
                                best_grid_parameters['precision']=float(precision)
                                best_grid_parameters['recall']=float(recall)
                                best_grid_parameters['loss']=float(loss.detach().item())
                                best_grid_parameters['val_loss']=float(val_loss)
                                best_grid_parameters['epochs']=int(epochs)
                                early_stopping(epoch, valid_loss, model , best_grid_parameters, True)
                                max_iou =  valid_dice

                                logger.info(msg)
                                # logger.info(f'best parameter-------{best_grid_parameters}')

                                # save the output to file
                                df_list = np.column_stack([epoch_list, train_epoch_loss, 
                                valid_epoch_loss, valid_epoch_precision, valid_epoch_recall, 
                                valid_epoch_accuracy, valid_epoch_dice])
                                df = pd.DataFrame(df_list, columns=["Epochs", "Training_loss", "Validation_loss", "Validation_precision",  "Validation_recall", 
                                "Validation_f1", "Validation_iou"])
        
                                df.to_csv(f'final_final_results/{filename}_Segmentation.csv', index=True) 
                            if early_stopping.early_stop:
                                print("Early stopping")
                                break
        logger.info(f'final best parameter-------{best_grid_parameters}----------')
        logger.info(f'best validation statistics-------{msg}-------------')
                            
    except KeyboardInterrupt:
        logger.warning('Training stopped manually!')

    
    logger.info(f'========= DONE ========')


def save_grid_parameters(gridsearch_space, lr_ixd, hs_ixd, filename):
    """
    save paramters from the grid-search for re-load.
    """
    current_random_parameters = {}
    learning_rate     = gridsearch_space['learning_rate']
    hidden_size       = gridsearch_space['hidden_size']
    save_path = f"final_final_results/{filename}_search_state.json"
    current_learning_rate = learning_rate[lr_ixd:]
    current_hidden_size   = hidden_size[hs_ixd:]
    current_random_parameters['learning_rate']=sorted(current_learning_rate)
    current_random_parameters['hidden_size']=sorted(current_hidden_size)
    json.dump(current_random_parameters, open(save_path, 'w'))
    logger.info(f'.....Saving and loading gridsearch parameters to {save_path}....')

def reshape_data(inputs, labels):
    """
    reshapes the data.
    """
    batch_size, num_channel, W, H = inputs.shape
    inputs = inputs.view(batch_size, H, num_channel* W)

    batch_size, num_channel, W, H = labels.shape
    labels = labels.view(batch_size, H, num_channel* W)

    return inputs, labels

def load_grid_parameters(gridsearch_space, filename):
    """
    Loads the check point of the gridsearch.
    """
    path  = f"final_final_results/{filename}_search_state.json"
    if os.path.exists(path): #check if file exist and load it
        logger.info(f'.....Gridsearch state file already exist {path}.......')
        with open(path) as f:
            parameters = json.load(f)
        
    else: # save new parameters
        # print('saving file and loading file')
        logger.info(f'.....Saving and loading grid state file.......')
        json.dump(gridsearch_space, open(path, 'w')) 
        with open(path) as f:
            parameters = json.load(f)
    
    return parameters



logger = get_logger(f'F:/MSC/Code/Python/logs/{filename}')
logger.info('Command-line arguments')
logger.info(parameters)

logger.info('.....Creating dataset.......')
# load training, validation and testing
trainloader         = dataloaders['train']
validloader         = dataloaders['val']

if __name__=='__main__':

    gridsearch_space_n = load_grid_parameters(gridsearch_space, filename)

    logger.info(f'Grid loaded {gridsearch_space}')

    model_hypersearch_fit(gridsearch_space_n, 
                    trainloader, 
                    validloader,
                    weighted_binary_cross_entropy,
                    image_dimension,
                    epochs,
                    filename
                                            )
