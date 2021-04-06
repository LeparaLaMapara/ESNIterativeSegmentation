import numpy as np

from tqdm import tqdm
import logging
import sys
import  os
from glob import glob


from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms.functional as TF


class LevelSetDataset(Dataset):
    """

    """
    def __init__(self, 
                input_image_path:str,
                target_image_path:str,
                threshold:float=0.5,
                num_past_steps:int=1,
                num_future_steps:int=1,
                num_frames:int=90,
                image_dimension:int=32,
                train_split:float=0.8,
                valid_split:float=0.1,
                training_mode:str='train'
                ):
        
        self.input_image_path    = input_image_path
        self.target_image_path   = target_image_path
        self.threshold           = threshold
        self.num_past_steps     = num_past_steps
        self.num_future_steps    = num_future_steps
        self.num_frames =num_frames
        self.image_dimension     = image_dimension
        self.valid_split         = valid_split
        self.train_split         = train_split
        self.training_mode       = training_mode
        
    

        # get a list of input filenames as sort them (e.g. 1.png, 2.png,..,N.png)
        self.input_image_fp = sorted(glob(os.path.join(self.input_image_path , "*")), 
                                    key=lambda x: int(os.path.basename(x).split('.')[0])
                                                     )
        print()
        print('leng', len(self.input_image_fp))
        print()
        if len(self.input_image_fp)>10000:
            n=10000
        else:
            n=len(self.input_image_fp)

        n=len(self.input_image_fp)
        ids = np.arange(1, n)
        # split the data
        train_idx  = int(self.train_split  * len(ids))
        valid_idx  = int(self.valid_split * len(ids))
        # ids_test = int(len(input_image_fp) - (id_train + ids_valid))

        # train
        if self.training_mode=='train':
            self.ids = ids[0:train_idx]
        #valid
        if self.training_mode=='valid':
            self.ids = ids[train_idx:train_idx+valid_idx]

        # test
        if self.training_mode=='test':
            self.ids = ids[train_idx+valid_idx:]
        
        # calculate mean and std using training set
        train_ids = ids[0:train_idx]
        train_fps = [self.input_image_fp[x] for x in train_ids]
        print(train_fps)

        self.mean_image   =  self._compute_mean(train_fps)
        self.stddev_image = self._compute_stddev(train_fps)

        self.transforms= torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(size=(self.image_dimension,self.image_dimension), 
                                                                        interpolation=Image.BILINEAR),
                                    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                    # torchvision.transforms.RandomVerticalFlip(p=0.5),
                                    torchvision.transforms.ToTensor()
                                    # ,torchvision.transforms.Normalize(mean=[self.mean_image], std=[self.stddev_image])
                                                                ])

    def _create_binary_mask(self, x):
        x[x>=self.threshold] = 1
        x[x <self.threshold] = 0
        return x

    def _stat_norm(self, x):
        norm =torchvision.transforms.Compose([torchvision.transforms.Resize(
            size=(self.image_dimension,self.image_dimension), 
                      interpolation=Image.BILINEAR),
                    torchvision.transforms.ToTensor()])
        return norm(x)

    def _compute_mean(self,  fp_list):
        mean_image = torch.zeros([1, self.image_dimension, self.image_dimension])
        file_counter = 0
        for fp in tqdm(fp_list, desc=f"Calculating mean image for {os.path.basename(fp_list[0]).split('/')[-1].split('/')[0]}"):
            mean_image+=self._stat_norm(Image.open(fp).convert('L'))   
            file_counter += 1
        mean_image /= file_counter
        return mean_image
        
    def _compute_stddev(self, fp_list):
        stddev_image = torch.zeros([1, self.image_dimension, self.image_dimension])
        file_counter = 0
        
        for fp in tqdm(fp_list, desc=f"Calculating stddev image for {os.path.basename(fp_list[0]).split('/')[-1].split('/')[0]}"):
            stddev_image += (self._stat_norm(Image.open(fp).convert('L')) - self.mean_image)**2
            file_counter += 1
        stddev_image /= file_counter
        stddev_image = torch.sqrt(stddev_image)
        return stddev_image

    def __len__(self):
        return len(self.ids) - (self.num_past_steps+self.num_future_steps)

    def _augs(self, x, seed):
        "https://pytorch.org/vision/stable/transforms.html"
        np.random.seed(seed)

        if np.random.random() > 0.5:
            x = TF.hflip(x)

        if np.random.random() > 0.5:
            x = TF.vflip(x)

        angles = [60, 90, 180, 270, 360]
        i = np.random.randint(0, len(angles))
        if np.random.random() > 0.5:
            x = TF.rotate(x, angles[i])

        # if np.random.random() > 0.5:
        #     x = TF.gaussian_blur(x, 3)
              # x = TF.affine()
        return x

    def __getitem__(self, index):
        if self.training_mode=='train':
            seed = np.random.randint(0, 100)
        else:
            seed=0
            
        X, Y , names = [], [], []
        idx = self.ids[index]
        xi = Image.open(os.path.join(self.input_image_path, str(self.ids[index])+'.jpg')).convert('L')
        xi = self.transforms(xi)
        xi  = (xi-self.mean_image)/(self.stddev_image)

        for step_idx, step in enumerate(np.arange(1, self.num_frames, self.num_past_steps)):
            x = Image.open(os.path.join(self.target_image_path,f'{idx}_{step}.jpg')).convert('L')
            x = self.transforms(x)
            if self.training_mode=='train':
                 xi  = self._augs(xi, seed)
                 x  = self._augs(x, seed)
            x = self._create_binary_mask(x)
            x = torch.stack((xi,x),dim=1)
            X.append(x)

        y = Image.open(os.path.join(self.target_image_path, f'{idx}_{(self.num_frames+self.num_past_steps+self.num_future_steps)-2}.jpg'))
        y = self.transforms(y)
        if self.training_mode=='train':
            y  = self._augs(y, seed)
        y = self._create_binary_mask(y)
        name  = f'{idx}_{self.num_frames-(self.num_past_steps+self.num_future_steps)}'
                                          
        X = torch.stack(X, dim=1)

        # (n, t, channel, h, w) ---> (n, channel, t, h, w)
        if len(X.shape)==5:
            X = X.permute(0,2,1,3,4)

        assert len(X.shape) == 5, "input shape is not a 5-dimensional array of (n, t, C, H, W)"
        assert len(y.shape) == 3, "output shape is not 3-dimensional  array of (C, H, W)"
        
        return X, y, name

    def create_set(self, batch_size ,shuffle=True,  pin_memory=True, num_workers=4, training_mode='train'):

        ds = LevelSetDataset(
        input_image_path=self.input_image_path,
        target_image_path=self.target_image_path,
        threshold=self.threshold,
        num_past_steps=self.num_past_steps,
        num_future_steps=self.num_future_steps,
        image_dimension=self.image_dimension,
        num_frames=self.num_frames ,
        valid_split= self.valid_split,     
        train_split= self.train_split,
        training_mode=training_mode
        )

        dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
        )
        return dl

