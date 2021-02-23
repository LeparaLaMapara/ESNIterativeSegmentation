import numpy as np

from tqdm import tqdm
import logging
import sys
import  os

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms.functional as TF

def lag_data(x, y, lag):
    """
    Shifts the time series with lag
    """
    x = x[:-lag]
    y = y[lag:]
    return x, y

class TimeSeriesImagesSegmentation(Dataset):
    def __init__(self,  original_image_path:str,  
                        segmentation_image_path:str, 
                        split:str='train', 
                        image_dimension:int=32,
                        window:int=1,
                        total_original_images:int=200, 
                        total_segemented_images:int=100,
                        threshold:float=0.5,
                        extentions:list=['.jpg','.jpg']):
        """
        Data loader for training on Shan-Vese generated images
        using full images. 
        """
        self.original_image_path     = original_image_path
        self.segmentation_image_path = segmentation_image_path

        if total_original_images  is None:
            # use all the images in the directory
             self.total_original_images  = len(os.listdir(original_image_path))
        else:
            self.total_original_images   = total_original_images

        self.total_segemented_images = total_segemented_images
        self.image_dimension         = image_dimension
        self.split                   = split
        self.extentions              = extentions
        self.threshold               = threshold
        self.window                  = window
        self.original_images, self.labels, self.names = self.get_file_names()   
        
    def __len__(self):
        return (self.total_original_images * self.total_segemented_images)

    def _transforms(self, image, mask, input_mask, testing_mask):
        
        resize     = torchvision.transforms.Resize(size=(self.image_dimension, self.image_dimension))
        image      = resize(image)
        input_mask = resize(input_mask)
        mask       = resize(mask)
        testing_mask      = resize(testing_mask)

        # if self.split=='train':
        #     # Random horizontal flipping
        #     if np.random.random() > 0.5:
        #         image      = TF.hflip(image)
        #         input_mask = TF.hflip(input_mask)
        #         mask       = TF.hflip(mask)

        #     # Random vertical flipping
        #     if np.random.random() > 0.5:
        #         image      = TF.hflip(image)
        #         input_mask = TF.hflip(input_mask)
        #         mask       = TF.hflip(mask)


#             if random.randon() > 0.5:
#                 image = TF.to_tensor(image) 
#                 image = image + torch.randn(image.size()) *1 +0
#             if random.random() > 0.5:
#                 image = TF.rotate(image, 45)
#                 label = TF.rotate(label, 45)

        # transform to tensor
        image        = TF.to_tensor(image) 
        input_mask   = TF.to_tensor(input_mask)
        mask         = TF.to_tensor(mask)
        testing_mask         = TF.to_tensor(testing_mask)

        # form image classes (i.e. background vs foreground)
        input_mask[input_mask>=self.threshold]   = 1
        input_mask[input_mask< self.threshold]   = 0

        mask[mask>=self.threshold]   = 1
        mask[mask< self.threshold]   = 0

        testing_mask[testing_mask>=self.threshold]   = 1
        testing_mask[testing_mask< self.threshold]   = 0
     
        if self.split=='train':
            # stack the images together (input + segmentation)
            stacked_images = torch.cat((image, input_mask), dim=0) 
        else:
            stacked_images =  torch.cat((image, testing_mask), dim=0) 
        # count and get the number of foreground pixels
        count_foreground_pixels =  (torch.nonzero(mask).size(0) / (self.image_dimension*self.image_dimension))
        return stacked_images, mask, count_foreground_pixels
    

    def get_file_names(self):
        """
        get the file name of the input and target images.
        """
        original_images, labels , names = [], [], []
        original_image_index = np.arange(1, len(os.listdir(self.original_image_path))+1)
        
        if self.split=='train':
            np.random.shuffle(original_image_index)
            
        original_image_index=original_image_index[:self.total_original_images]
        for original_image in original_image_index: # for each original image
            for segmentation_image in np.arange(1, self.total_segemented_images+1): # for each segmentation image
                image_name_original      =  str(original_image) +                                      self.extentions[0]
                image_name_segementation =  str(original_image) + '_' +  str(segmentation_image)   +   self.extentions[1]

                # store the names of the input images
                original_images.append(image_name_original)
                labels.append(image_name_segementation)
                names.append(self.original_image_path + '/' + image_name_original)
        return original_images, labels, names

    def __getitem__(self, index):
        # inputs (t-1)
        # print(self.original_image_path, self.original_images[index:index+self.window][0])
        image              = Image.open(self.original_image_path     + '/' + self.original_images[index]).convert('L')  
        input_mask         = Image.open(self.segmentation_image_path + '/' + self.labels[index])   
        # labels (t)
        mask        = Image.open(self.segmentation_image_path + '/' + self.labels[index])   
        testing_mask = Image.open(self.segmentation_image_path + '/' + self.labels[0])   
        # print(image.shape, label.shape)
        
         # get image name
        image_name = str(self.names[index].split("/")[-1])
        
        # apply data augmentation if train and transformations
        stacked_images, labels, count_foreground_pixels  = self._transforms(image, mask, input_mask, testing_mask)
      
        return stacked_images, labels, image_name, count_foreground_pixels

def fetch_dataloader(parameters):
    dataloaders = {}
    for partition in tqdm(['train','val','test']):
        original_images_path     =  parameters['original_image_path']     + partition
        segmentation_images_path =  parameters['segmentation_image_path'] + partition 
        print('Loading {} dataset'.format(partition))
        if partition =='train':
            mass_roads_dataset= TimeSeriesImagesSegmentation(original_image_path=original_images_path, 
            segmentation_image_path=segmentation_images_path,
            split=partition,
            window=parameters['window'],
            image_dimension=parameters['image_dimension'],
            total_original_images=parameters['total_original_images'],
            total_segemented_images=parameters['total_segmentation_images'],
            threshold=parameters['threshold'],
            extentions=parameters['extentions']
            )
            dl = torch.utils.data.DataLoader(mass_roads_dataset, batch_size=parameters['batch_size'],
            shuffle=parameters['shuffle'] ,
            num_workers=parameters['num_workers'],
            pin_memory=parameters['pin_memory'])                                     
        else:
            mass_roads_dataset       = TimeSeriesImagesSegmentation(original_image_path=original_images_path, 
            segmentation_image_path=segmentation_images_path,
            split=partition,
            window=parameters['window'],                                                        
            image_dimension=parameters['image_dimension'],
            total_original_images=parameters['total_original_images'],
            total_segemented_images=parameters['total_segmentation_images'],
              threshold=parameters['threshold'],
            extentions=parameters['extentions']
            )
            dl = torch.utils.data.DataLoader(mass_roads_dataset, batch_size=1, 
            shuffle=False,
            num_workers=parameters['num_workers'],
            pin_memory=parameters['pin_memory'])

        dataloaders[partition] = dl
    return dataloaders

