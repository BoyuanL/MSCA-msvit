import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
import math

# 导入所需模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from pretrain_models.vit_face import ViT_face
from pretrain_models.vit_face_enhanced import EnhancedViT_face
from pretrain_models.ViTAE_Reduction_Cell import msvit

class DualPathBackbone(nn.Module):
    
    def __init__(self, backbone="vit", pretrain=True):
        super().__init__()
        
        # 模型配置参数
        self.image_size = 112
        self.patch_size = 8
        self.dim = 512
        self.depth = 20
        self.heads = 8
        self.mlp_dim = 2048
        self.dropout = 0.1
        self.emb_dropout = 0.1
        
        # 路径1：冻结的原始ViT_face (教师模型)
        self.original_encoder = ViT_face(
            loss_type='ArcFace',
            GPU_ID=0,
            num_class=93431,
            image_size=self.image_size,
            patch_size=self.patch_size,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            emb_dropout=self.emb_dropout
        )
        
        # 路径2：增强型ViT_face (学生模型)
        self.enhanced_encoder = EnhancedViT_face(
            loss_type='ArcFace',
            GPU_ID=0,
            num_class=93431,
            image_size=self.image_size,
            patch_size=self.patch_size,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            emb_dropout=self.emb_dropout
        )
        
        # 多尺度特征提取器
        self.ms_extractor = msvit(
            img_size=self.image_size,
            in_chans=3,
            embed_dims=32,  # 较小的维度减少参数量
            token_dims=64,  # 输出特征维度
            downsample_ratios=2,  # 较小的下采样率保留更多空间信息
            kernel_size=5,
            dilations=[1, 2, 3],  # 多尺度感受野
            tokens_type='performer',  # 轻量级注意力
            op='sum'  # 使用sum而非cat降低复杂度
        )
        
        # 特征投影层
        self.projection = nn.Linear(64, self.dim)  # 投影到与ViT特征相同维度
        self.alpha = nn.Parameter(torch.ones(1) * 2.0)  # sigmoid(2.0) ≈ 0.88
        self.training_stage = 0
        
        # 加载预训练权重
        if pretrain:
            try:
                # 加载权重文件
                pretrained_path = os.path.join(parent_dir, "pretrain_models", 
                                             "Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth")
                check_load = torch.load(pretrained_path, map_location='cpu')
                
                # 为原始编码器加载权重并冻结
                self.original_encoder.load_state_dict(check_load, strict=True)
                for param in self.original_encoder.parameters():
                    param.requires_grad = False
                print("成功加载预训练权重到原始ViT编码器")
                
                # 为增强型编码器加载权重
                model_dict = self.enhanced_encoder.state_dict()
                filtered_dict = {k: v for k, v in check_load.items() if k in model_dict}
                model_dict.update(filtered_dict)
                self.enhanced_encoder.load_state_dict(model_dict, strict=False)
                print(f"成功加载 {len(filtered_dict)}/{len(model_dict)} 预训练参数到增强型编码器")
                
            except Exception as e:
                print(f"加载预训练模型时发生错误: {e}")
        
        # 设置输出特征维度
        self.backbone_feature = self.dim
        self.imagesize = self.image_size
        
        # 默认冻结所有增强编码器参数
        self._freeze_enhanced_encoder()
        # 只允许训练msvit, 投影层和alpha参数
        self._unfreeze_specific_layers(['ms_extractor', 'projection', 'alpha'])
        
        print(f"DualPathBackbone 初始化完成: dim={self.dim}, image_size={self.image_size}")
    
    def _freeze_enhanced_encoder(self):
        """冻结增强型编码器所有参数"""
        for param in self.enhanced_encoder.parameters():
            param.requires_grad = False
            
    def _unfreeze_specific_layers(self, layer_keywords):
        """解冻包含指定关键字的层"""
        for name, param in self.named_parameters():
            for keyword in layer_keywords:
                if keyword in name:
                    param.requires_grad = True
                    break
    
    def set_training_stage(self, stage):
        """设置训练阶段，调整参数可训练状态"""
        self.training_stage = stage
        for param in self.parameters():
            param.requires_grad = False
        for param in self.original_encoder.parameters():
            param.requires_grad = False
        if stage >= 1:
            for name, param in self.named_parameters():
                if any(keyword in name for keyword in ['ms_extractor', 'projection', 'alpha']):
                    param.requires_grad = True

        if stage >= 2:
            for name, param in self.enhanced_encoder.named_parameters():
                if any(f'transformer.layers.{i}' in name for i in [17, 18, 19]):
                    param.requires_grad = True

        if stage >= 3:
            for param in self.enhanced_encoder.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"训练阶段 {stage}: 可训练参数数量 = {trainable_params:,}")
    
    def get_trainable_parameters(self):
        """返回当前阶段可训练的参数"""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def forward(self, img):
        with torch.no_grad():
            original_emb = self.original_encoder(img)

        ms_features = self.ms_extractor(img)

        ms_features_proj = self.projection(ms_features)

        enhanced_emb = self.enhanced_encoder(img, ms_features=ms_features_proj)

        weight = torch.sigmoid(self.alpha)
        final_emb = weight * original_emb + (1 - weight) * enhanced_emb
        
        # 返回L2归一化后的特征
        return F.normalize(final_emb)