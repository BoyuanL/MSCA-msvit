import os
import argparse
import torch
from main import training  # 导入原始训练函数

def main():
    
    parser = argparse.ArgumentParser(description="分阶段训练")
    parser.add_argument("--relation", type=str, default="bb", 
                        choices=['ss', 'bb', 'sibs', 'md', 'fs', 'ms', 'fd', 'gmgd', 'gfgd', 'gmgs', 'gfgs'],
                        help="亲属关系类型")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tau", default=0.08, type=float)
    parser.add_argument("--optimizer", type=str, choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()
    
    # 设置固定参数
    args.backbone = "msvit"
    args.staged_training = True

    print("\n" + "="*50)
    print("="*50)
    args.stage = 1
    args.epochs = 5
    args.lr = 1e-4
    training(args)

    print("\n" + "="*50)
    print("="*50)
    args.stage = 2
    args.epochs = 5
    args.lr = 5e-5
    training(args)

    print("\n" + "="*50)
    print("="*50)
    args.stage = 3
    args.epochs = 10
    args.lr = 1e-5
    training(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()