
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter
import torch
import numpy as np
import utils
import os
class pose_eff_dataset(Dataset):
   def __init__(self, data_dir,data,transform=None):
     self.imageFileRel=data["image"]
     self.data_dir=data_dir
     self.yaw=data["yaw"]
     self.pitch=data["pitch"]
     self.roll=data["roll"]
     self.x_min = data["x_min"]
     self.y_min = data["y_min"]
     self.x_max = data["x_max"]
     self.y_max = data["y_max"]
     self.length = len(data)
     self.transform =transform
   def __getitem__(self,index):
        img = Image.open(os.path.join(
            self.data_dir,self.imageFileRel[index] + '.jpg'))
        img = img.convert('RGB')
        img = img.crop((int(self.x_min[index]), int(self.y_min[index]), int(self.x_max[index]), int(self.y_max[index])))
        rnd = np.random.random_sample()
        if rnd < 0.5:
            self.yaw[index] = -self.yaw[index]
            self.roll[index] = -self.roll[index]
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)
        R = utils.get_R(self.pitch[index], self.yaw[index], self.roll[index])

        if self.transform is not None:
            img = self.transform(img)

        return img,  torch.FloatTensor(R)
   def __len__(self):
        # 12,450
        return self.length
        
class BIWI(Dataset):
    def __init__(self, filename_path, transform, image_mode='RGB', train_mode=True):
        #self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        #cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        #labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        #cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R)
        #, cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length
