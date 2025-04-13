import os
import sys
import torch
# 获取当前文件所在目录的父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将父目录加入到模块搜索路径中
sys.path.append(parent_dir)
from pretrain_models.vit_face import ViT_face
# from pretrain_models.vit_face_RC_NC import ViT_face
from pretrain_models.vits_face import ViTs_face
import torch.nn.functional as F
import torchvision.models as models
# 导入增强版backbone
# 修改后 - 使用绝对导入
try:
    from code1.backbones_enhanced import MSAttentionBackbone
except ImportError:
    try:
        from backbones_enhanced import MSAttentionBackbone
    except ImportError:
        print("无法导入 MSAttentionBackbone，请确保 backbones_enhanced.py 文件存在")
# 导入新的双路径backbone
from code1.backbones_dual_path import DualPathBackbone
class Backbone(torch.nn.Module):   # 用于图像特征提取任务

    def __init__(self,
                 backbone="vit",
                 pretrain=True,
                 ):
        # 初始化函数，接收两个参数，backbone表示使用哪个backbone模型，pretrain表示是否使用预训练的backbone模型。
        super(Backbone, self).__init__()
        assert backbone in ['vit','vits','msvit'] # 断言backbone的取值
        self.backbone=backbone  # 保存backbone参数的取值。

        if self.backbone=="vit":
            self.encoder= ViT_face(  loss_type ='ArcFace',
                         GPU_ID = 0,
                         num_class = 93431,
                         image_size=112,  # 112
                         patch_size=8,
                         dim=512,
                         depth=20,
                         heads=8, # 8   /   1
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1)

            # 加载预训练模型
            check_load = torch.load("../pretrain_models/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth")
            # check_load = {'encoder.' + k: v for k, v in check_load.items()} # 修改
            self.encoder.load_state_dict(check_load,strict=True)
            self.backbone_feature=512 # 特征向量的维度为512。
            self.imagesize=112



            print("load pretrained ViT_face model")


        elif self.backbone=='vits':
            self.encoder = ViTs_face(loss_type='ArcFace',
                         GPU_ID=0,
                         num_class=93431,
                         image_size=224,   # image_size=112
                         patch_size=16, #8
                         ac_patch_size=12,
                         pad = 4,
                         dim=512,
                         depth=20,
                         heads=8,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1)
                # if pretrain else ViTs_face()
            self.backbone_feature = 2048
            self.imagesize=224
            check_load = torch.load("../pretrain_models/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth")

            self.encoder.load_state_dict(check_load,strict=False)
            print("load pretrained ViTs_face model")

        
        elif self.backbone=='msvit':
            # 使用双路径backbone
            self.encoder = DualPathBackbone(backbone="vit", pretrain=pretrain)
            print('test:使用双路径backbone使用双路径backbone使用双路径backbone使用双路径backbone')
            self.backbone_feature = self.encoder.backbone_feature
            self.imagesize = self.encoder.imagesize
            print("使用双路径多尺度ViT模型")
            # # 使用MSAttentionBackbone作为编码器
            # self.encoder = MSAttentionBackbone(backbone="vit", pretrain=pretrain)
            # self.backbone_feature = self.encoder.backbone_feature  # 512
            # self.imagesize = self.encoder.imagesize  # 112
            # print("使用多尺度注意力增强ViT模型")

    def forward(self, img):
        # 前向计算函数，接收一个输入img，经过self.encoder的处理后得到特征向量emb，再对emb进行归一化处理后返回。
        emb=self.encoder(img)
        return F.normalize(emb)



