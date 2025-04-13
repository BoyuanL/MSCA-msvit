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
        self.target_layer = self.model.encoder.transformer.layers[0][1].fn.norm

        # 图像预处理参数
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.transform = tfs.Compose([
            tfs.Resize((112, 112)),
            tfs.ToTensor(),
            tfs.Normalize(self.means, self.stds)
        ])

    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        return result.transpose(2, 3).transpose(1, 2)

    def process_img(self, input):
        return self.transform(input).unsqueeze(0)

    def _generate_mask(self, grayscale_cam):
        """优化后的掩码生成函数"""
        heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)

        # 1. 动态阈值（均值+2倍标准差）
        mean_val = np.mean(heatmap_uint8)
        std_val = np.std(heatmap_uint8)
        thresh = int(mean_val + 2 * std_val)
        _, binary = cv2.threshold(heatmap_uint8, max(thresh, 220), 255, cv2.THRESH_BINARY)

        # 2. 小核形态学处理（椭圆核更平滑）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. 轮廓处理（凸包简化+面积过滤）
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(processed)

        if contours:
            # 按面积过滤（最小面积阈值设为图像面积的0.3%）
            max_contour = max(contours, key=cv2.contourArea)
            min_area = 112 * 112 * 0.003  # 约38像素

            if cv2.contourArea(max_contour) > min_area:
                # 生成凸包并填充
                convex_hull = cv2.convexHull(max_contour)
                cv2.drawContours(mask, [convex_hull], -1, 255, -1)

                # 4. 强化腐蚀（使用更小的核）
                erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.erode(mask, erode_kernel, iterations=2)

        return mask

    def __call__(self, img_root):
        # 预处理
        img_pil = Image.open(img_root).convert("RGB").resize((112, 112))
        input_tensor = self.process_img(img_pil)

        # 生成热力图
        cam = GradCAM(model=self.model,
                      target_layers=[self.target_layer],
                      use_cuda=torch.cuda.is_available(),
                      reshape_transform=self.reshape_transform)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        # 生成并应用掩码
        mask = self._generate_mask(grayscale_cam)
        original_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)

        # 保存结果
        cv2.imwrite('test_masked_result.jpg', masked_img)

        # 保存调试中间结果
        cv2.imwrite('test_heatmap_raw.jpg', (grayscale_cam * 255).astype(np.uint8))
        cv2.imwrite('test_mask_debug.jpg', mask)

        # 生成热力图叠加
        visualization = show_cam_on_image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(float) / 255,
                                          grayscale_cam,
                                          use_rgb=True)
        cv2.imwrite('test_heatmap_overlay.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    cam_vit = CalCAM_ViT()
    img_root = "../data/FIW/Train/train-faces/F0232/MID2/P02467_face0.jpg"
    cam_vit(img_root)