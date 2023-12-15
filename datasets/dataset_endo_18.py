import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, H, W, transform = True):
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

        # 空间变换应用于图像和标签
        self.spatial_transform = A.Compose([
            #A.RandomResizedCrop(H, W, scale=(0.08, 1.0)),
            A.Flip(p=0.75),
            A.RandomRotate90()
        ])

        # 像素级变换 - 只应用于图像
        self.pixel_transform = A.Compose([
            A.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ])

        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        filename = self.sample_list[idx].strip('\n')
        image_path = os.path.join(self.data_dir, self.split, "images", filename + '.png')
        label_path = os.path.join(self.data_dir, self.split, "annotations/instrument", filename + '.png')

        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))

        if self.transform:
            # 先应用空间变换
            transformed = self.spatial_transform(image=image, mask=label)
            image, label = transformed['image'], transformed['mask']

            # 再应用像素级变换
            transformed_image = self.pixel_transform(image=image)
            image = transformed_image['image']
        '''
        print(image.shape)
        print(label.shape)
        print(image)
        print(label)
        '''
        label = label[:, :, 0]
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # Change from HxWxC to CxHxW
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label, 'case_name': filename}

if __name__ == "__main__":
    import cv2

    def save_overlayed_image(dataset, idx, class_colors, save_path="example_2.png"):
        sample = dataset[idx]
        image = sample['image'].numpy().transpose(1, 2, 0)  # Convert CxHxW to HxWxC

        # 检查 label 是否已经是 Tensor，如果是，就不需要再转换
        if isinstance(sample['label'], torch.Tensor):
            label = sample['label'].numpy()
        else:
            label = sample['label']

        # 将标签映射到颜色
        colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for cls_idx, cls_color in class_colors.items():
            colored_label[label == cls_idx] = cls_color

        # 叠加图像和标签
        overlayed = 0.7 * image + 0.3 * colored_label
        overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

        # 保存图像
        cv2.imwrite(save_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))


    # 类别到颜色的映射
    class_colors = {
        1: [255, 0, 0],  # 类别 1 映射到红色
        2: [0, 255, 0],  # 类别 2 映射到绿色
        3: [0, 0, 255],  # 类别 3 映射到蓝色
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
        7: [100, 255, 0] # 类别 7 映射到黄色
    }

    dataset = Synapse_dataset(base_dir='/hy-tmp/I157378211f00901317/ISINet_Train_Val/resized', list_dir='/hy-tmp/I157378211f00901317/SAMed/lists/lists_Endovis18', split='train', H=1024, W=1024)

    # 可视化第一个样本
    save_overlayed_image(dataset, 0, class_colors)
