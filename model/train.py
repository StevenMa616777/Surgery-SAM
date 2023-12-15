import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

# 假设您的模型、数据集和损失函数类已经定义好
from surgery_sam import SurgerySAM
import sys
sys.path.append("../datasets")
from dataset_endo_18 import Synapse_dataset
from Loss_func import SAMLoss

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
'''
# 参数设置
num_epochs = 200
learning_rate = 1e-5
batch_size = 8
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_steps = 0
warmup_steps = 500 
num_classes = 7
test = False


# 数据集和数据加载器
train_dataset = Synapse_dataset(base_dir='./ISINet_Train_Val/resized', list_dir='./lists/lists_Endovis18', split='train', H=1024, W=1024)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
'''

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Training parameters.')


parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
parser.add_argument('--test_mode', action='store_true', help='Run in test mode')
parser.add_argument('--base_dir', type=str, default='./ISINet_Train_Val/resized', help='Base directory for the dataset')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Endovis18', help='Directory for the lists')
parser.add_argument('--checkpoint_path', type=str, default='/hy-tmp/I157378211f00901317/SAMed/checkpoints/sam_vit_b_01ec64.pth', help='Path to the model checkpoint')
parser.add_argument('--tensorboard_log_dir', type=str, default='/tf_logs/runs/exp_2', help='Directory for TensorBoard logs')
parser.add_argument('--checkpoint_save_dir', type=str, default='/hy-tmp/checkpoint(rank=4)/', help='Directory where to save checkpoints')

# 解析参数
args = parser.parse_args()

# 使用解析得到的参数
num_epochs = args.num_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
warmup_steps = args.warmup_steps
num_classes = args.num_classes
test = args.test_mode
base_dir = args.base_dir
list_dir = args.list_dir
checkpoint_path = args.checkpoint_path
tensorboard_log_dir = args.tensorboard_log_dir
checkpoint_save_dir = args.checkpoint_save_dir

# 数据集和数据加载器
train_dataset = Synapse_dataset(base_dir=base_dir, list_dir=list_dir, split='train', H=1024, W=1024)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = SurgerySAM(
        model_type="vit_b",
        checkpoint=checkpoint_path,  # 模型权重文件的路径
        prompt_dim=256,
        num_classes=num_classes + 1,
        r = 4,  # LoRA rank
        lora_layers=None,
        freeze_image_encoder = True,
        freeze_prompt_encoder = True,
        freeze_mask_decoder = False,
        mask_HW = (1024, 1024),
        feature_input = False,
        prompt_decoder = False,
        dense_prompt_decoder=False,
        no_sam=False
    ).cuda()

def adjust_learning_rate(optimizer, step, warmup_steps, initial_lr, decay_rate=0.97, decay_steps=1000):
    """根据步骤数调整学习率，包括 warmup 和 decay"""
    if step < warmup_steps:
        lr = initial_lr * step / warmup_steps
    else:
        lr = initial_lr * (decay_rate ** (step / decay_steps))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
def save_combined_image(image_tensor, mask_tensor, pred_mask_tensor, epoch, output_dir='outputs'):
    """将原始图像和预测掩码组合并保存为图片"""

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 转换图像和掩码张量为 PIL 图像
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image_tensor)
    mask_pil = to_pil(mask_tensor.float())  # 将 mask 转换为浮点数，以便转换为 PIL 图像
    pred_mask_pil = to_pil(pred_mask_tensor.float())

    # 将图像、真实掩码和预测掩码水平堆叠
    combined_image = Image.new('RGB', (image_pil.width * 3, image_pil.height))
    combined_image.paste(image_pil, (0, 0))
    combined_image.paste(mask_pil, (image_pil.width, 0))
    combined_image.paste(pred_mask_pil, (image_pil.width * 2, 0))

    # 保存图像
    output_path = os.path.join(output_dir, f'output_epoch_{epoch}.png')
    combined_image.save(output_path)
    '''

# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = SAMLoss(focal_cof = 0.4, dice_cof = 0.6, ce_cof = 0.,  iou_cof = 0.06).cuda()

writer = SummaryWriter(tensorboard_log_dir)



# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    train_loader = tqdm(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].cuda()
        labels = batch['label'].cuda()
        #print(images.shape)
        #print(labels.shape)
        if test:
            break
        # 前向传播
        pred_masks, ious = model(images)
        if isinstance(pred_masks, list):
            pred_masks = torch.stack(pred_masks, dim=0)
        #print(pred_masks.shape)
        if isinstance(ious, list):
            ious = torch.stack(ious, dim=0)
        #print(ious.shape)
        loss_dict = loss_fn(pred_masks, labels.long(), ious)
        loss = loss_dict['loss']

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adjust_learning_rate(optimizer, total_steps, warmup_steps, learning_rate)
        total_steps += 1
        
        total_loss += loss.item()
        train_loader.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    model_save_path = os.path.join(checkpoint_save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)

    '''
    with torch.no_grad():
        idx = random.randint(0, len(train_loader.dataset) - 1)
        sample = train_loader.dataset[idx]
        image, mask = sample['image'].unsqueeze(0).cuda(), sample['label'].unsqueeze(0).cuda()
        pred_masks, _ = model(image)
        if isinstance(pred_masks, list):
            pred_masks = torch.stack(pred_masks, dim=0)
        # 选择一个类别的掩码作为示例
        pred_mask = pred_masks[0, 0]  # 假设选择第一个类别

        save_combined_image(image[0], mask[0], pred_mask, epoch)
    '''

writer.close()

print("Training completed.")
