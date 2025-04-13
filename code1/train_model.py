import os
import random
import numpy as np
import torch.multiprocessing as get_logger
import torch.nn
from sklearn.metrics import roc_curve
from datasets import ContrastiveTrain, ValAndTest
from torch.optim import SGD,Adam
from losses import *
import argparse
from torch.utils.data import DataLoader
from files import Dir
from code1.backbones11 import Backbone
from utils import *
import warnings

# 新增标签列表
relationlist = ['ss','md','fs','ms','fd','sibs','bb','gmgd','gfgd']

# def read_data(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip().split('\t')
#             # 修改读入的数据，将标签放在最后一个元素
#             data.append((line[:-1], line[-1]))
#     return data
#
# def to_one_hot(tag):
#     # 从标签列表中获取标签的索引
#     tag_index = relationlist.index(tag)
#     # 将标签转换为one-hot向量
#     one_hot = np.zeros(len(relationlist))
#     one_hot[tag_index] = 1
#     return one_hot
#
# def get_data(file_path):
#     data = read_data(file_path)
#     # 将标签转换为one-hot向量
#     x = np.array([np.array(list(map(int, item[0]))) for item in data])
#     y = np.array([to_one_hot(item[1]) for item in data])
#     return x, y
#
# def get_model(input_shape, output_shape, dropout=0.2):
#     inputs = Input(shape=input_shape)
#     x = LSTM(128)(inputs)
#     x = Dropout(dropout)(x)
#     x = Dense(128)(x)
#     x = Dropout(dropout)(x)
#     outputs = Dense(output_shape, activation='softmax')(x)
#     model = Model(inputs, outputs)
#     return model
#
# def train(file_path):
#     x, y = get_data(file_path)
#     input_shape = (x.shape[1], 1)
#     output_shape = len(relationlist) # 输出的维度改为len(relationlist)
#     model = get_model(input_shape, output_shape)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#     model.fit(x.reshape((x.shape[0], x.shape[1], 1)), y, epochs=500, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
#     return model

def training(args):

    batch_size = args.batch_size
    val_batch_size = args.batch_size

    epochs = args.epochs
    train_steps = args.train_steps
    backbone = args.backbone
    tau = args.tau

    save_dir = Dir.mkdir("infonce_"+backbone)

    log_path = os.path.join(save_dir,"train.log")
    logger = get_logger(log_path)

    model = Backbone(backbone=backbone).cuda()
    ckpt_path = './pretrain_models/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    train_dataset = ContrastiveTrain(sample_path="../data/FIW/pairs/train.txt",
                                     backbone=backbone,
                                     images_size=model.imagesize
                                     )

    val_dataset = ValAndTest(sample_path='../data/FIW/pairs/val_choose.txt',
                             backbone=backbone,
                             images_size=model.imagesize
                             )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=0, pin_memory=False)


    if args.optimizer=='sgd':
        optimizer_model = SGD(model.parameters(),lr=1e-4,momentum=0.9)
    else:
        optimizer_model = Adam(model.parameters(),lr=1e-5)


    contrastive=ContrastiveLoss().cuda()

    max_acc=0.0

    for epoch_i in range(1,epochs+1):

        logger.info('epoch ' + str(epoch_i ))
        contrastive_loss_epoch = []

        model.train()
        for index_i, data in enumerate(train_loader):
            img1, img2,kin_class,label = data

            emb1 = model(img1)
            emb2 = model(img2)

            loss=contrastive(emb1,emb2,tau=tau)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            contrastive_loss_epoch.append(loss.item())

            if index_i  == train_steps-1:
                break

        use_sample=epoch_i*batch_size*train_steps
        train_dataset.set_bias(use_sample)

        logger.info("infonce_loss:" + "%.6f" % np.mean(contrastive_loss_epoch))

        model.eval()
        acc,threshold = val_model(model, val_loader)

        logger.info("acc is %.6f " % acc)
        logger.info("threshold is %.6f " % threshold)

        if max_acc < acc:
            logger.info("validation acc improve from :" + "%.6f" % max_acc + " to %.6f" % acc)
            max_acc = acc
            save_model(model, os.path.join(save_dir, "infonce_"+backbone+"_best_model.pkl"))
        else:
            logger.info("validation acc did not improve from %.6f" % float(max_acc))

    save_model(model, os.path.join(save_dir, "infonce_"+backbone+"_final_model.pkl"))

def save_model(model, path):
    torch.save(model.encoder.state_dict(), path)



@torch.no_grad()
def val_model(model, val_loader):
    y_true = []
    y_pred = []

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
    # conf = get_conf(y_pred, threshold)
    # ece = ECELoss().cuda()(conf,acc).item()

    return torch.mean(acc).item(), threshold
    # return torch.mean(acc).item(),ece,threshold



if __name__ == '__main__':
    file_path = 'data.txt'
    # 原先是从标签列表中随机选取一个标签作为要训练的标签，现在改为遍历训练所有标签
    for relation in relationlist:
        print(f'Training model for relation {relation}...')
        # 修改获取训练数据的函数，只获取特定标签的数据
        x, y = get_data(file_path, relation)
        input_shape = (x.shape[1], 1)
        output_shape = len(relationlist)
        model = get_model(input_shape, output_shape)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(x.reshape((x.shape[0], x.shape[1], 1)), y, epochs=500, batch_size=128, validation_split=0.2,
                  callbacks=[early_stopping])

    warnings.filterwarnings('ignore',category=UserWarning)
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--batch_size", type=int, default=10)  # default=25
    parser.add_argument("--epochs", type=int, default=80)  # default=80
    parser.add_argument("--train_steps", type=int, default=20, help="number of iterations per epoch")  # default=50

    parser.add_argument("--tau", default=0.08, type=float, help="infoNCE temperature parameters")

    # parser.add_argument("--backbone", type=str, choices=['resnet50', 'resnet101'], default="resnet101")
    parser.add_argument("--backbone", type=str, choices=['vit','vits'], default="vit")
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    # parser.add_argument("-head", "--head", help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",default='ArcFace', type=str)
    # parser.add_argument("--loss", default='Softmax', type=str, help="loss-type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)


