import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 获取当前文件所在目录的父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将父目录加入到模块搜索路径中
sys.path.append(parent_dir)
# 导入修改后的ViT_face和原始损失函数
from pretrain_models.vit_face_enhanced import EnhancedViT_face
from pretrain_models.ViTAE_Reduction_Cell import msvit, MSCE 

class MSAttentionBackbone(nn.Module):
    def __init__(self, backbone="vit", pretrain=True):
        super().__init__()
        
        # ViT参数配置
        image_size = 112
        patch_size = 8
        dim = 512
        depth = 20
        heads = 8
        mlp_dim = 2048
        dropout = 0.1
        emb_dropout = 0.1
        
        self.encoder = EnhancedViT_face(
            loss_type='ArcFace',
            GPU_ID=0,
            num_class=93431,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        self.ms_extractor = msvit(
            img_size=image_size,
            in_chans=3,
            embed_dims=128, # 64
            token_dims=512,  # msvitCell输出特征维度   128
            downsample_ratios=4,  # 下采样率
            kernel_size=7,
            dilations=[1,2,3,4],  # 多尺度膨胀率
            tokens_type='performer',  # 使用performer注意力
            op='cat'  # 特征融合方式
        )
        
        if hasattr(self.ms_extractor, 'MSCE') and hasattr(self.ms_extractor.MSCE, 'out_chans'):
            ms_feature_dim = self.ms_extractor.MSCE.out_chans
        else:
            ms_feature_dim = 128  # 默认值，根据实际情况调整
            
        # 特征投影层 - 将特征投影到与ViT_face兼容的维度
        self.projection = nn.Linear(ms_feature_dim, dim)       
        # 加载预训练权重
        if pretrain:
            try:
                pretrained_path = "../pretrain_models/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth"
                check_load = torch.load(pretrained_path, map_location='cpu')
                # 获取模型的state_dict
                model_state_dict = self.encoder.state_dict()
                # 过滤权重，只保留名称匹配的参数
                filtered_dict = {}
                for k, v in check_load.items():
                    # 尝试直接加载
                    if k in model_state_dict:
                        filtered_dict[k] = v
                    # 尝试添加"encoder."前缀
                    elif f"encoder.{k}" in model_state_dict:
                        filtered_dict[f"encoder.{k}"] = v
                    # 如果仍然不匹配，则跳过
                # 使用strict=False加载预训练权重
                missing, unexpected = self.encoder.load_state_dict(filtered_dict, strict=False)
            except Exception as e:
                print(f"加载预训练模型失败: {e}")
                # 继续使用随机初始化的权重
        # 设置输出特征维度和图像尺寸
        self.backbone_feature = dim  # 输出特征维度
        self.imagesize = image_size  # 输入图像尺寸
    
    def forward(self, img):
        ms_features = self.ms_extractor(img)  # [B, N, C]
        ms_features_proj = self.projection(ms_features)  # [B, N, dim]
        emb = self.encoder(img, ms_features=ms_features_proj)
        # 4. 应用L2归一化并返回
        return F.normalize(emb)