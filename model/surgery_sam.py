# Surgery_SAM.py


import copy
from functools import reduce
from operator import mul
import math

import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from einops import rearrange
import torch
import torch.nn as nn
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d
#from prompt_encoder import Prototype_Prompt_Encoder, Learnable_Prototypes

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class SurgerySAM(nn.Module):
    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "/hy-tmp/I157378211f00901317/SAMed/checkpoints/sam_vit_b_01ec64.pth",
            pixel_mean = [118.38626424,  86.92436099, 92.46854535],
            pixel_std = [51.71419209, 47.71345123, 50.68595686],
            prompt_dim: int = 256,
            num_classes: int = 7,
            r: int = 4,  # LoRA rank
            lora_layers=None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False,
    ):
        super().__init__()

        self.model = sam_model_registry[model_type](checkpoint=checkpoint, pixel_mean=pixel_mean, pixel_std=pixel_std )

        #self.sam_model_with_lora = LoRA_Sam(self.model, r, lora_layers)

        if lora_layers is None:
            lora_layers = list(range(len(self.model.image_encoder.blocks)))
        self.apply_lora_to_image_encoder(r, lora_layers)
        
        self.mask_HW = mask_HW
        self.feature_input = feature_input

        # 修改掩码提示
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
                for i in range(self.model.mask_decoder.num_mask_tokens)
            ]
        )

        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
                                                                           self.model.mask_decoder.num_mask_tokens)
        '''
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        '''
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def apply_lora_to_image_encoder(self, r, lora_layers):
        """应用 LoRA 修改到 image encoder."""
        # 冻结 image encoder 参数
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        # 对每个 block 应用 LoRA
        for layer_i, block in enumerate(self.model.image_encoder.blocks):
            if layer_i in lora_layers:
                self.apply_lora_to_block(block, r)

    def apply_lora_to_block(self, block, r):
        """在特定 block 中应用 LoRA 修改."""
        # 获取 qkv 层
        qkv = block.attn.qkv
        dim = qkv.in_features

        # 创建 LoRA 层
        linear_a_q = nn.Linear(dim, r, bias=False)
        linear_b_q = nn.Linear(r, dim, bias=False)
        linear_a_v = nn.Linear(dim, r, bias=False)
        linear_b_v = nn.Linear(r, dim, bias=False)

        # 替换 qkv 层
        block.attn.qkv = _LoRA_qkv(
            qkv,
            linear_a_q,
            linear_b_q,
            linear_a_v,
            linear_b_v,
        )
    
    def forward(self, images):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(images)

        pred_masks = []
        ious = []
        for embedding in image_embeddings:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            if self.dense_prompt_decoder is not None:
                embedding_img = embedding.flatten(1).permute(1, 0)
                sparse_embeddings_v = self.model.mask_decoder.mask_tokens.weight.clone()
                org_shape = dense_embeddings.shape
                dense_embeddings_gen = self.dense_prompt_decoder(embedding_img, sparse_embeddings_v)
                dense_embeddings_gen = dense_embeddings_gen.permute(1, 0).reshape(*org_shape)
                dense_embeddings = dense_embeddings + dense_embeddings_gen

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(0))
            ious.append(iou_predictions.reshape(-1, 1))

        return pred_masks, ious



# 下面是实例化模型并测试的部分
if __name__ == "__main__":
    
    import torch

    # 假设模型输入是1024x1024的RGB图像
    batch_size = 8  # 用于测试的批量大小
    num_channels = 3  # RGB图像有3个通道
    height, width = 1024, 1024  # 输入图像的尺寸

    # 创建一个随机图像张量作为测试数据
    test_images = torch.rand(batch_size, num_channels, height, width).cuda()

    # 确保替换以下参数以匹配您的模型配置
    model = SurgerySAM(
        model_type="vit_b",
        checkpoint="/hy-tmp/I157378211f00901317/SAMed/checkpoints/sam_vit_b_01ec64.pth",  # 模型权重文件的路径
        prompt_dim=256,
        num_classes=7,
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

    model.eval()

    # 使用测试数据进行前向传播
    with torch.no_grad():  # 不需要计算梯度
        output = model(test_images)

    # 检查输出的类型和形状
    print(type(output))
    for item in output:
        print(type(item))
        if isinstance(item, torch.Tensor):
            print(item.shape)
        elif isinstance(item, list):
            print([type(subitem) for subitem in item])
            print([subitem.shape if isinstance(subitem, torch.Tensor) else "Not a tensor" for subitem in item])

    #print([o.shape for o in output])
