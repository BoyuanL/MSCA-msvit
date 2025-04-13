import os
#from multiprocessing import get_logger
import numpy as np
import torch.nn
from collections import OrderedDict
# 导入get_logger函数
from Utils import get_logger
from sam import SAM
from datasets import ContrastiveTrain, ContrastiveVal, ContrastiveTest
from torch.optim import SGD,Adam
from losses import *
import argparse
from torch.utils.data import DataLoader
from files import Dir
from backbones11 import Backbone
from utils import *
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from backbones_dual_path import DualPathBackbone

ckpt_dir = 'C:/Users/24673/PostGraduate/Bproject/DEMO3-0428/pretrain_models/' #  ../pretrain_models/
ckpt_name = 'Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'   # Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_family_ids(samples_list):
    family_ids = []
    for sample in samples_list:
        # 假设样本的路径结构固定，如 'Validation/val-faces/F0446/MID2/P04714_face1.jpg'
        # 使用 split 来解析出 family_id
        parts = sample[0].split('/')  # 这里使用'/'分割路径
        family_id = parts[2]  # 取出代表 family_id 的部分，例如 F0446
        family_ids.append(family_id)
    return family_ids
def training(args):

    batch_size = args.batch_size
    val_batch_size = args.batch_size

    epochs = args.epochs
    train_steps = args.train_steps
    backbone = args.backbone
    tau = args.tau
    relation = args.relation
    optimizer = args.optimizer
    save_dir = Dir.mkdir("infonce_"+backbone)
    lr = args.lr
    aug = args.aug

    subfolder_name = relation + "_" + str(batch_size) + "_" + str(epochs) + "_" + str(train_steps) + "_" + optimizer + "_" + str(lr) + "_" + aug

    save_dir_subfolder = os.path.join(save_dir, subfolder_name)
    os.makedirs(save_dir_subfolder, exist_ok=True)
    log_path = os.path.join(save_dir_subfolder, "train.log")
    logger = get_logger(log_path)
    model = Backbone(backbone=backbone).cuda()
    train_dataset = ContrastiveTrain(sample_path="../data/FIW/pairs/train.txt",
                                     backbone=backbone,
                                     images_size=model.imagesize,
                                     relation=relation,
                                     rand_mirror=True
                                     )

    val_dataset = ContrastiveVal(sample_path='../data/FIW/pairs/val_choose.txt',
                             backbone=backbone,
                             images_size=model.imagesize,
                             relation = relation
                             )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=0, pin_memory=False)

    # 获取familyID
    # 获取val_loader里的samples_list
    samples_list = val_loader.dataset.samples_list
    family_ids = extract_family_ids(samples_list)
    if args.backbone == 'msvit' and hasattr(model.encoder, 'set_training_stage'):
        model.encoder.set_training_stage(args.stage)
        # 使用特定的优化器
        if args.optimizer == 'adam':
            optimizer_model = Adam(model.encoder.get_trainable_parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer_model = SGD(model.encoder.get_trainable_parameters(), lr=args.lr, momentum=0.9)
        else:
            base_optimizer = torch.optim.Adam
            optimizer_model = SAM(model.encoder.get_trainable_parameters(), base_optimizer, lr=0.1)
    else:
        # 原始优化器代码不变
        if args.optimizer=='sgd':
            optimizer_model = SGD(model.parameters(),lr=1e-6,momentum=0.9)
        elif args.optimizer=='adam':
            optimizer_model = Adam(model.parameters(),lr=1e-6)
        else:
            base_optimizer = torch.optim.Adam  
            optimizer_model = SAM(model.parameters(), base_optimizer, lr=0.1)


    contrastive=ContrastiveLoss().cuda()
    best_model = None #保存最佳模型
    max_acc=0.0

    logger.info('starting: ' + relation)
    logger.info('batch_size: ' + str(batch_size))
    logger.info('epochs:' + str(epochs))
    logger.info('train_steps:' + str(train_steps))
    logger.info('optimizer: ' + optimizer +'    lr: '+ str(lr))
    logger.info('aug: ' + aug)

    for epoch_i in range(1,epochs+1):
        print('-' * 40)
        logger.info('epoch ' + str(epoch_i))
        contrastive_loss_epoch = []

        model.train()
        for index_i, data in enumerate(train_loader):
            img1, img2,kin_class,label = data

            emb1 = model(img1.to(device))
            emb2 = model(img2.to(device))
            loss=contrastive(emb1,emb2,tau=tau)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            contrastive_loss_epoch.append(loss.item())

            if index_i  == train_steps-1:
                break

        use_sample=epoch_i*batch_size*train_steps


        logger.info("infonce_loss:" + "%.6f" % np.mean(contrastive_loss_epoch))

        model.eval()
        val_acc,threshold,_,_,_ = val_model(model, val_loader)


        logger.info("val_acc is %.6f " % val_acc)
        logger.info("threshold is %.6f " % threshold)
        if max_acc < val_acc:
            logger.info("validation acc improve from :" + "%.6f" % max_acc + " to %.6f" % val_acc)
            max_acc = val_acc
            best_model = model.state_dict()  # 保存最佳模型权重
            save_model(model,
                       os.path.join(save_dir_subfolder, f"{backbone}_Epoch{epoch_i}_{val_acc:.6f}_best_model.pth"))
        else:
            logger.info("validation acc did not improve from %.6f" % float(max_acc))

    # 使用最佳模型绘制最终的 ROC 曲线
    if best_model is not None:
        model.load_state_dict(best_model)  # 加载最佳模型
        logger.info("Loaded best model for ROC curve")
        val_acc,_,roc_auc,fpr,tpr = val_model(model, val_loader)

        logger.info(f"Final Validation Accuracy: {val_acc:.6f}, AUC: {roc_auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'SS')  

        # 添加随机猜测基线
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

        # 设置图表标题、坐标轴和图例
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curve for FIW', fontsize=14)
        plt.legend(loc='lower right', fontsize=12)  # 显示图例
        plt.grid(alpha=0.4)

        # plt.show()
    logger.info("threshold before calibration: %.6f" % threshold)


def save_model(model, path):
    torch.save(model.encoder.state_dict(), path)



@torch.no_grad()
def val_model(model, val_loader):
    y_true = []
    y_pred = []
    samples_list = val_loader.dataset.samples_list
    for img1, img2, kin_class,labels in val_loader:
        e1 = model(img1.cuda())
        e2 = model(img2.cuda())
        pred=torch.cosine_similarity(e1, e2, dim=1)
        y_pred.append(pred)
        y_true.append(labels)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    fpr, tpr, thresholds_keras = roc_curve(y_true.view(-1).cpu().numpy().tolist(),
                                           y_pred.view(-1).cpu().numpy().tolist())
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    acc = ((y_pred >= threshold).float() == y_true).float()
    # 计算 AUC
    roc_auc = auc(fpr, tpr)
    return torch.mean(acc).item(), threshold,roc_auc,fpr,tpr  # 返回准确率、阈值及提取的特征

@torch.no_grad()
def test(model, test_loader):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img1, img2, kin_class, label in test_loader:
            e1 = model(img1.cuda())
            e2 = model(img2.cuda())
            pred = torch.cosine_similarity(e1, e2, dim=1)
            y_pred.append(pred)
            y_true.append(label)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    fpr, tpr, thresholds_keras = roc_curve(y_true.view(-1).cpu().numpy().tolist(),
                                           y_pred.view(-1).cpu().numpy().tolist())
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    acc_before = ((y_pred >= threshold).float() == y_true).float()
    return acc_before, threshold

def load_model(backbone='vit', ckpt_path=ckpt_dir + ckpt_name):
    model = Backbone(backbone=backbone).to(device)
    ckpt_path = r'C:\Users\24673\PostGraduate\Aproject\DEMO3-0428\pretrain_models\Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'
    check_load = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    check_load = {'encoder.' + k: v for k, v in check_load.items()} # 修改
    model.load_state_dict(check_load, strict=True)
    model.eval()
    return model
if __name__ == '__main__':
    warnings.filterwarnings('ignore',category=UserWarning)
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=2, help="number of iterations per epoch")
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate (default: 1e-5)')
    parser.add_argument("--tau", default=0.08, type=float, help="infoNCE temperature parameters")
    parser.add_argument("--backbone", type=str, choices=['vit','vits','msvit'], default="msvit")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--aug",type=str,choices=['flip','flip-dataIncrease'],default="flip")
    relation = parser.add_argument("--relation", type=str,choices=['ss', 'bb', 'sibs', 'md', 'fs',
                                                                   'ms', 'fd', 'gmgd', 'gfgd','gmgs' ,'gfgs'], default="bb")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    # 添加新的参数
    parser.add_argument("--staged_training", action="store_true")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    # set_seed(seed=100)
    training(args)

