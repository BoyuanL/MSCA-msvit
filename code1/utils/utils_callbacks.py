import logging
import os
import time
from typing import List
from sklearn.metrics import roc_curve
import torch

# from .eval import verification
# from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from torch.utils.data import DataLoader
from code1.eval import verification
# from code1.main2 import save_model
from code1.utils.utils_logging import AverageMeter


class CallBack_KinVerification(object):
    def __init__(self, val_dataset, backbone, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0, pin_memory=False)  # 假设验证批次大小为32
        self.highest_acc: float = 0.0
        self.backbone = backbone
        self.summary_writer = summary_writer

    def ver_test(self, global_step: int):
        y_true = []
        y_pred = []
        self.backbone.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for img1, img2, kin_class, label in self.val_loader:
                e1 = self.backbone(img1.cuda())
                e2 = self.backbone(img2.cuda())

                e1 = torch.tensor(e1[0])
                e2 = torch.tensor(e2[0])
                pred = torch.cosine_similarity(e1, e2, dim=1)

                y_pred.append(pred)
                y_true.append(label)

            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)

            fpr, tpr, thresholds_keras = roc_curve(y_true.view(-1).cpu().numpy().tolist(),
                                                   y_pred.view(-1).cpu().numpy().tolist())

            maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
            threshold = thresholds_keras[maxindex]

            acc = ((y_pred >= threshold).float() == y_true).float()
            val_acc,threshold = torch.mean(acc).item(), threshold
                # predictions = outputs > 0.5  # 假设阈值为0.5
                # total_correct += (predictions == label).sum().item()
                # total_samples += label.size(0)
        # accuracy = total_correct / total_samples
        logging.info('[%d] Accuracy: %f' % (global_step, val_acc))   # val_acc?
        logging.info("threshold is %.6f " % threshold)
        # if self.summary_writer:
        #     self.summary_writer.add_scalar('Accuracy', acc, global_step)
        #
        # if acc > self.highest_acc:
        #     self.highest_acc = acc
        # logging.info('[%d] Highest Accuracy: %f' % (global_step, self.highest_acc))

        #-----------------
        if self.highest_acc < val_acc:
            logging.info("validation acc improve from :" + "%.6f" % self.highest_acc + " to %.6f" % val_acc)
            max_acc = val_acc
            # save_model(self.backbone, os.path.join(save_dir_subfolder, self.backbone + "_Epoch" + str(epoch_i) + "-" + str(epochs) + "_acc%.6f" % val_acc + "_best_model.pkl"))
        else:
            logging.info("validation acc did not improve from %.6f" % float(self.highest_acc))

    def __call__(self, num_update):
        if self.rank == 0 and num_update > 0:
            self.ver_test(num_update)
            self.backbone.train()


# class CallBackVerification(object):
#
#     def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
#         self.rank: int = distributed.get_rank()
#         self.highest_acc: float = 0.0
#         self.highest_acc_list: List[float] = [0.0] * len(val_targets)
#         self.ver_list: List[object] = []
#         self.ver_name_list: List[str] = []
#         if self.rank is 0:
#             self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
#
#         self.summary_writer = summary_writer
#
#     def ver_test(self, backbone: torch.nn.Module, global_step: int):
#         results = []
#         for i in range(len(self.ver_list)):
#             acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
#                 self.ver_list[i], backbone, 10, 10)
#             logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
#             logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
#
#             self.summary_writer: SummaryWriter
#             self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )
#
#             if acc2 > self.highest_acc_list[i]:
#                 self.highest_acc_list[i] = acc2
#             logging.info(
#                 '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
#             results.append(acc2)
#
#     def init_dataset(self, val_targets, data_dir, image_size):
#         for name in val_targets:
#             path = os.path.join(data_dir, name + ".bin")
#             if os.path.exists(path):
#                 data_set = verification.load_bin(path, image_size)
#                 self.ver_list.append(data_set)
#                 self.ver_name_list.append(name)
#
#     def __call__(self, num_update, backbone: torch.nn.Module):
#         if self.rank is 0 and num_update > 0:
#             backbone.eval()
#             self.ver_test(backbone, num_update)
#             backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
