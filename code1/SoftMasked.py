from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
                             XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
from code1.main import load_model  # 从 main.py 中导入 load_model 函数
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torchvision.transforms as tfs

class CalCAM_ViT(nn.Module):
    def __init__(self):
        super(CalCAM_ViT, self).__init__()
        self.model = load_model()  # 加载预训练模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(self.model)
        # 用于 Grad-CAM 的目标层
        # self.target_layer = self.model.encoder.transformer.layers[-1][1].fn.norm
        self.target_layer = self.model.encoder.transformer.layers[0][1].fn.norm

        # 图像预处理参数
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]

        self.transform = tfs.Compose([
            tfs.Resize((112, 112)),  # 修改图像预处理步骤以确保图像大小一致
            tfs.ToTensor(),
            tfs.Normalize(self.means, self.stds)
        ])

    def reshape_transform(self, tensor, height=14, width=14):
        # 去掉cls token
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def process_img(self, input):
        input = self.transform(input)
        input = input.unsqueeze(0)
        return input

    def __call__(self, img_root):
        img = Image.open(img_root)
        img = img.resize((112, 112))
        plt.imshow(img)
        plt.savefig("vit_input_md_m_ori.jpg")
        input = self.process_img(img)

        cam = GradCAM(model=self.model, target_layers=[self.target_layer], use_cuda=torch.cuda.is_available(),
                      reshape_transform=self.reshape_transform)

        grayscale_cam = cam(input_tensor=input, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # 将 grad-cam 的输出叠加到原始图像上
        img = np.array(img)[:, :, ::-1]  # 转换为 BGR
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255, grayscale_cam)

        # 保存可视化结果
        # cv2.imwrite('vit_img_md_m.jpg', visualization)
        # --- 新增：生成软掩码并应用 ---
        # 1. 将热力图转为uint8
        heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)

        # 2. 将热力图归一化到0-1范围
        heatmap_normalized = heatmap_uint8 / 255.0

        # 3. 将软掩码应用到原图
        original_img = np.array(img)[:, :, ::-1]  # PIL转OpenCV BGR格式
        masked_img = original_img * heatmap_normalized[:, :, np.newaxis]

        # 保存结果
        cv2.imwrite('masked_result_soft_mdm.jpg', masked_img)

        # 原热力图叠加（可选）
        visualization = show_cam_on_image(original_img.astype(float) / 255, grayscale_cam)
        cv2.imwrite('heatmap_overlay_soft_mdm.jpg', visualization)

if __name__ == "__main__":
    cam_vit = CalCAM_ViT()
    img_root = "../data/FIW/Train/train-faces/F0232/MID2/P02467_face0.jpg"  # 请替换为你的图像路径
    # data/FIW/Train/train-faces/F0001/MID4/P00008_face3.jpg # son
    # data/FIW/Train/train-faces/F0001/MID1/P00001_face0.jpg # father
    # data/FIW/Train/train-faces/F0187/MID3/P02004_face1.jpg fd_d
    # data/FIW/Train/train-faces/F0102/MID3/P01051_face2.jpg ms_m
    # data/FIW/Train/train-faces/F0102/MID1/P01050_face1.jpg ms_s
    # data/FIW/Train/train-faces/F0232/MID2/P02467_face0.jpg md_m
    # data/FIW/Train/train-faces/F0232/MID4/P02464_face30.jpg md_d
    cam_vit(img_root)