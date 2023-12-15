import os
import os.path as osp
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T
import random
import torch

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "/hy-tmp/I157378211f00901317/ISINet_Train_Val/surgical/", 
                 mode = "val", 
                 vit_mode = "h", transform = 0):
        
        self.data_root_dir = data_root_dir 
        self.mode = mode 
        self.vit_mode = vit_mode
        self.transform = transform
        
        self.mask_dir = osp.join(self.data_root_dir, self.mode, "binary_annotations")
        self.mask_list = []
        self.image_dir = osp.join(self.data_root_dir, self.mode, "images")

        for subdir, _, files in os.walk(self.mask_dir):
            if '.ipynb_checkpoints' in subdir:
                continue
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir), i) for i in files]
        #print(len(self.mask_list))
        if self.mode == "train":
            #self.image_dir = osp.join(self.data_root_dir, self.mode, "images")
                
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(1024, scale=(0.8, 1.0)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                #T.ToTensor()
            ])
            # To convert the mask to single channel with values 0 or 255
            self.mask_transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(1024, scale=(0.8, 1.0)),
                T.RandomRotation(10),
                #T.Grayscale(),
                #T.ToTensor()
            ])

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        #print(mask_name)
        # get class id from mask_name 
        cls_id = int(re.search(r"class_?(\d+)", mask_name).group(1))
        
        if self.mode == "train":
            # Extract sequence and frame info from mask_name
            

            seq_info_match = re.search(r"seq_\d+", mask_name)
            if seq_info_match:
                seq_info = seq_info_match.group(0)
                #print(seq_info)
            else:
                raise ValueError(f"Cannot extract seq info from mask name {mask_name}")

            # 提取 frame_info
            frame_info_match = re.search(r"(\d{5})_class", mask_name)
            if frame_info_match:
                frame_info = frame_info_match.group(1)[-3:]
            else:
                raise ValueError(f"Cannot extract frame info from mask name {mask_name}")
            # 使用 seq_info 和 frame_info 构造 image_name
            image_name = f"{seq_info}_frame{frame_info}.png"
            image_path = os.path.join(self.data_root_dir, self.mode, "images", image_name)
            image_path = osp.join(self.image_dir, image_name)
            image = Image.open(image_path)    
    
            # 加载mask
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path)

            if self.transform:
                seed = random.randint(0, 2147483647)  # Generate a random seed
                np.random.seed(seed)  # Set numpy random seed
                image = self.transforms(image)
                np.random.seed(seed)  # Reset numpy random seed
                mask = self.mask_transforms(mask)

            image_np = np.array(image)
            mask_np = np.array(mask)
            #print(mask_np.shape)
            mask_np = mask_np[:,:,0]
            #image = to_tensor(image_np)
            #mask = to_tensor(mask_np)
            image = torch.from_numpy(image_np).permute(2, 0, 1)  # Change from HxWxC to CxHxW
            mask = torch.from_numpy(mask_np)
            mask = mask.float() / 255.0
            image = image.float()

            
            return image, mask, cls_id
        else:
            seq_info_match = re.search(r"seq_\d+", mask_name)
            if seq_info_match:
                seq_info = seq_info_match.group(0)
                #print(seq_info)
            else:
                raise ValueError(f"Cannot extract seq info from mask name {mask_name}")

            # 提取 frame_info
            frame_info_match = re.search(r"(\d{5})_class", mask_name)
            if frame_info_match:
                frame_info = frame_info_match.group(1)[-3:]
            else:
                raise ValueError(f"Cannot extract frame info from mask name {mask_name}")
            # 使用 seq_info 和 frame_info 构造 image_name
            image_name = f"{seq_info}_frame{frame_info}.png"
            image_path = os.path.join(self.data_root_dir, self.mode, "images", image_name)
            image_path = osp.join(self.image_dir, image_name)
            image = Image.open(image_path)    
    
            # 加载mask
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path)
            
            image_np = np.array(image)
            mask_np = np.array(mask)
            #print(mask_np.shape)
            mask_np = mask_np[:,:,0]
            #image = to_tensor(image_np)
            #mask = to_tensor(mask_np)
            image = torch.from_numpy(image_np).permute(2, 0, 1)  # Change from HxWxC to CxHxW
            mask = torch.from_numpy(mask_np)
            mask = mask.float() / 255.0
            image = image.float()

            
            return image, mask_name, cls_id
'''
from torch.utils.data import DataLoader

# 实例化数据集
dataset = Endovis18Dataset(mode = "val", vit_mode = "h", transform = 0)

# 定义batch size
batch_size = 32

# 实例化data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 迭代data loader
for batch in dataloader:
    images, masks, cls_ids = batch
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Cls_ids shape:", cls_ids.shape)
'''
'''
import numpy as np
import torch
from torch.utils.data import DataLoader

batch_size = 16  # 或者根据你的内存大小调整

def compute_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.var(2).sum(0)
        total_images_count += batch_samples
    
    mean /= total_images_count
    std = torch.sqrt(std / total_images_count)
    
    return mean, std


# 实例化数据集
dataset = Endovis18Dataset(mode="train", transform=True)

# 计算平均值和标准差
mean, std = compute_mean_std(dataset)
print('Mean:', mean)
print('Std:', std)
'''