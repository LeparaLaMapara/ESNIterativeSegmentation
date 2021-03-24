from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pandas as pd
import argparse

import logging    
from utils.logger import get_logger
from tqdm import tqdm
from glob import glob

from torch.utils import tensorboard
import json, os, sys
import time

import errno
import os

import matplotlib.pyplot as plt


sys.path.append('/')
from utils.utils import LevelSetDataset
from utils.metrics import iou_pytorch, pixel_segementation_evaluation 
from utils.PytorchEarlyStopping import EarlyStopping
from utils.loss import weighted_binary_cross_entropy, generalised_loss
from models.ConvRNN import CRNN, CESN
from models.Conv3D import    CNN3D

if __name__=="__main__":

    # parse augments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, help="Name of the run [CNN3D]")
    parser.add_argument("--data-path", type=str, required=True, help="Path to where the data is located")
    parser.add_argument("--save-path", type=str, required=True, help="Path to where runs will be written")
    parser.add_argument("--dataset", type=str, required=True, help="Path to where runs will be written")


    parser.add_argument("--num-epochs", type=int, default=500, help="Number of training epochs [500]")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of examples to use in a batch [32]")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for training [0.1]")

    parser.add_argument("--num-frames", type=int, default=95, help="Length of the sequences for each image [95]")
    parser.add_argument("--num-past-step", type=int, default=1, help="Number of steps to use in input [1]")
    parser.add_argument("--num-future-step", type=int, default=1, help="Number of time steps in the future for predictions [1]")
    parser.add_argument("--image-dimension", type=int, default=32, help="Dimensions to resize the images [32]")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pixel cutoff to create mask [0.5]")


    parser.add_argument("--in-channels", type=int, default=3, help="Input channel for the 1st conv layer [3]")
    parser.add_argument("--hidden",  type=int, default=512, help="Number of hidden units in the 1st  fully connected layer [512]")
    parser.add_argument("--num-classes", type=int, default=1024, help="Number of pixel classes to be predicted [1024]")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the recurrent unit [1]")
#     parser.add_argument("--sample-size", type=int, default=128 , help=" [128]")
#     parser.add_argument("--sample-duration", type=int, default=16, help=" [16]")
    parser.add_argument("--leaking-rate", type=float, default=0.01, help="Leak rate for leaky ESN [0.01]")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="Scaling of reservoir matrix [0.9]")
    parser.add_argument("--sparsity", type=float, default=0.2, help="Percentage of neurons with zeros in the reservoir matrix [0.2]")
    
#     parser.add_argument("--in-channels", type=int, default=3, help="Input channel for the 1st conv layer [3]")
    # parser.add_argument("--hidden-one",  type=int, default=512, help="Number of hidden units in the 1st  fully connected layer [512]")
    # parser.add_argument("--hidden-two",  type=int, default=256, help="Number of hidden units in the 2ndt  fully connected layer [256]")
#     parser.add_argument("--num-classes", type=int, default=1024, help="Number of pixel classes to be predicted [1024]")
    parser.add_argument("--dropout-prob", type=float, default=0.5, help="Dropout probability [0.5]")
    # parser.add_argument("--sample-size", type=int, default=128 , help=" [128]")
    # parser.add_argument("--sample-duration", type=int, default=16, help=" [16]")
    
#     parser.add_argument("--in-channels", type=int, default=3, help="Input channel for the 1st conv layer [3]")
#     parser.add_argument("--hidden",  type=int, default=512, help="Number of hidden units in the 1st  fully connected layer [512]")
#     parser.add_argument("--num-classes", type=int, default=1024, help="Number of pixel classes to be predicted [1024]")
#     parser.add_argument("--num-layers", type=int, default=1, help="Number of layers in the recurrent unit [1]")
    parser.add_argument("--sample-size", type=int, default=128 , help=" [128]")
    parser.add_argument("--sample-duration", type=int, default=16, help=" [16]")
    # parser.add_argument("--rnn-unit", type=str, default='LSTM', required=True, help="Recurrent unit type [LSTM]")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation [0]")



    args = parser.parse_args()
    run_path = os.path.join( args.save_path, args.run_name)

    # logging
    logger = get_logger(run_path)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get the number of classess
    args.num_classes = args.image_dimension*args.image_dimension

    # log all parameters
    logger.info("Commnad-line arguments")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"arguments {args}: {value}")
    
     # data loader
    logger.info("Creating dataset......")
    ls_dataset = LevelSetDataset(
        input_image_path=os.path.join(args.data_path,"images"),
        target_image_path=os.path.join(args.data_path,"labels"),
        threshold=args.threshold,
        num_past_steps=args.num_past_step,
        num_future_steps=args.num_future_step,
        image_dimension=args.image_dimension,
         num_frames=args.num_frames ,
        valid_split= 0.1,     
        train_split= 0.8,
        training_mode='train'
        )

    # device to perform computation (CPU or GPU)
    device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
     # create test ds for final evaluation
    logger.info(f"Creating test dataset for evaluating best model......")
    ls_eval_ds = ls_dataset.create_set(batch_size=1, shuffle=True, pin_memory=True, num_workers=4, training_mode='test')

    # CNN
    model_cnn = CNN3D(
    in_channels=args.in_channels,
    sample_size=args.sample_size,
    sample_duration=args.sample_duration,
    drop_p=args.dropout_prob, 
    hidden1=args.hidden,
    hidden2=args.hidden,
    num_classes=args.num_classes
    ).to(device)
    
    # LSTM
    model_lstm = CRNN(
    in_channels=args.in_channels,
    sample_size=args.sample_size,
    sample_duration=args.sample_duration,
    hidden_size=args.hidden,
    num_layers=args.num_layers,
    rnn_unit='LSTM',
    num_classes=args.num_classes
    ).to(device)
    
        
    # GRU
    model_gru = CRNN(
    in_channels=args.in_channels,
    sample_size=args.sample_size,
    sample_duration=args.sample_duration,
    hidden_size=args.hidden,
    num_layers=args.num_layers,
    rnn_unit='GRU',
    num_classes=args.num_classes
    ).to(device)
    
    # RNN
    model_rnn = CRNN(
    in_channels=args.in_channels,
    sample_size=args.sample_size,
    sample_duration=args.sample_duration,
    hidden_size=args.hidden,
    num_layers=args.num_layers,
    rnn_unit='RNN',
    num_classes=args.num_classes
    ).to(device)
    
    
    # ESN
    model_esn = CESN(
    in_channels=args.in_channels,
    sample_size=args.sample_size,
    sample_duration=args.sample_duration,
    hidden_size=args.hidden,
    num_layers=args.num_layers,
    num_classes=args.num_classes,
    leaking_rate=args.leaking_rate,
    spectral_radius=args.spectral_radius,
    sparsity=args.sparsity
    ).to(device)
    
    models ={
        'convESN':model_esn,
        'convLSTM':model_lstm,
        'convGRU':model_gru,
        'convRNN':model_rnn,
        'conv3d':model_cnn
    }
    logger.info(f"model: {models}")
    
    for model_key in models:
        logger.info(f"model name:{model_key}")
        for ds in [args.dataset]:
            logger.info(f"model: {ds}")
            model= models[model_key]
            fp=f'/home-mscluster/tmashinini/MSC/Data/processed_data/{ds}/results/*{ds}-{model_key}*/checkpoints/*.pt'
            logger.info(f"fp: {fp}")

            model_fp = glob(fp)[0]





            logger.info(f"model: {model_fp}")

            logger.info(f"Loading model for testing.....")
            model.load_state_dict(torch.load(model_fp))

            model.eval()
            for batch_idx, (inputs, labels, names) in enumerate(ls_eval_ds):
                                    
                # load data and move data to GPU's
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs= inputs.squeeze(1)

                # forward-propogation
                outputs = model(inputs) 
                outputs = (outputs >= args.threshold)*1

                outputs = outputs.view(-1, args.image_dimension, args.image_dimension)
                labels = labels.view(-1, args.image_dimension, args.image_dimension)
                inputs = inputs.to(device, non_blocking=True)
                # logger.info('shapes', inputs.shape, outputs.shape, labels.shape)
                # logger.info('shapes slice', inputs[0,0,0].shape, outputs[0].shape, labels[0].shape)

                # save the prediction and the labels
                # for t in np.arange(inputs.shape[2], 10):
                #     label = labels[:, :, t]
                #     output = outputs[:, :, t]
                t=81
                cp=model_fp.split('/')[-3]
                logger.info(f"save cp: {cp}")
                save_path  = f'/home-mscluster/tmashinini/MSC/Data/processed_data/{ds}/results/{cp}/predictions/'
                logger.info(f"save path: {save_path}")

                os.makedirs(save_path, exist_ok=True)

                
                plt.imsave(os.path.join(save_path,f'input.png'), inputs.detach().cpu().numpy()[0,0,0] )
                plt.imsave(os.path.join(save_path,f'label_{t}.png'), labels.detach().cpu().numpy()[0] )
                plt.imsave(os.path.join(save_path,f'output_{t}.png'), outputs.detach().cpu().numpy()[0] )

                break


        logger.info(f'========= DONE ========')
