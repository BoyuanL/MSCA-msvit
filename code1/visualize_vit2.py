from code1.main import load_model
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np
import cv2


class CalCAM_ViT(nn.Module):
    def __init__(self):
        super(CalCAM_ViT, self).__init__()
        self.model = load_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # 修改目标层为实际特征输出层（例如Transformer层输出）
        self.target_layer = self.model.encoder.transformer.layers[0][1].fn

        # 注册钩子捕获特征
        self.feature_maps = None
        self._register_hook()

        # 图像预处理
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.transform = tfs.Compose([
            tfs.Resize((112, 112)),
            tfs.ToTensor(),
            tfs.Normalize(self.means, self.stds)
        ])

    def _register_hook(self):
        def hook_function(module, input, output):
            self.feature_maps = output.detach()

        self.target_layer.register_forward_hook(hook_function)

    def reshape_transform(self, tensor, height=14, width=14):
        # 处理Transformer的序列输出
        result = tensor[:, 1:, :]  # 去掉cls token
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # [B, C, H, W]
        return result

    def process_img(self, input):
        input = self.transform(input)
        return input.unsqueeze(0).to(self.device)

    def visualize_feature(self, feature, img_size=112):
        # 处理特征图
        if len(feature.shape) == 4:  # [B, C, H, W]
            feature = feature.mean(dim=1)  # 通道维度取平均

        # 归一化
        feature = feature.squeeze().cpu().numpy()
        feature = (feature - feature.min()) / (feature.max() - feature.min())

        # 上采样到原图尺寸
        return cv2.resize(feature, (img_size, img_size))

    def __call__(self, img_root):
        # 处理输入图像
        img = Image.open(img_root).convert('RGB')
        img_resized = img.resize((112, 112))
        input_tensor = self.process_img(img_resized)

        # 前向传播获取特征
        with torch.no_grad():
            _ = self.model(input_tensor)

        # 检查是否捕获到特征
        if self.feature_maps is None:
            raise RuntimeError("未捕获到特征图，请检查钩子注册")

        # 转换特征图
        processed_feature = self.reshape_transform(self.feature_maps)
        feature_map = self.visualize_feature(processed_feature)

        # 可视化设置
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 显示原图
        axes[0].imshow(img_resized)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        # 显示特征图
        axes[1].imshow(feature_map, cmap='viridis')
        axes[1].axis('off')
        axes[1].set_title('Feature Map')

        # 保存结果
        plt.savefig('feature_visualization_fs_f.jpg', bbox_inches='tight')
        plt.close()

        # 返回特征图数据
        return feature_map


# 使用示例
if __name__ == "__main__":
    visualizer = CalCAM_ViT()
    feature_map = visualizer("../data/FIW/Train/train-faces/F0001/MID1/P00001_face0.jpg")
    # data/FIW/Train/train-faces/F0232/MID2/P02467_face0.jpg md_m
    # data/FIW/Train/train-faces/F0001/MID1/P00001_face0.jpg fs_f