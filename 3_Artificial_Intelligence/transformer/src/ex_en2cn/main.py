import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataLoader
from arguments import get_args

from _model.transformer import Transformer
from _common.util import *


class Runner:

    def __init__(self, args):
        # 1. 初始化参数
        self.args = args
        # 2. 创建数据集加载器，模型
        self.loader = DataLoader(args)
        self.model = Transformer(args)
        self.model.apply(he_initialization)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lr_decay, eps=args.adam_eps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.src_pad_idx)
        # 3. 实验结果记录
        now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.save_path = os.path.join(get_abs_path(root=None, rel_path="ex_en2cn"), args.save_dir, now)
        self.save_path_model = os.path.join(self.save_path, args.save_dir_model)
        self.writer = SummaryWriter(self.save_path)
        os.makedirs(self.save_path_model, exist_ok=True)
        log_args(args, path=self.save_path)
        self.batch_count = 0
    
    def train(self):
        # 0. 开启训练模式
        print("----------------------Train----------------------")
        self.model.train()
        # 1. 加载训练数据（train.source.cn.txt，train.target.en.txt）
        X, Y = self.loader.load_train_data()
        # 2. 训练 n_epoch 轮，每轮
        for epoch in range(self.args.n_epoch):
            for index, _ in self.loader.get_batch_indices(len(X), args.batch_size):
                # 2.1. 获取一批数据
                x_batch = torch.LongTensor(X[index]).to(args.device)
                y_batch = torch.LongTensor(Y[index]).to(args.device)
                # 2.2. 前向传播，计算 Loss
                self.optimizer.zero_grad()
                predict_batch = self.model(x_batch, y_batch)
                loss = self.criterion(input=predict_batch.contiguous().view(-1, predict_batch.shape[-1]), target=y_batch.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
                # 2.3. 记录日志
                self.batch_count += 1
                self.writer.add_scalar("Loss", loss.detach().cpu().numpy(), self.batch_count)
            # 2.4. 保存模型
            if epoch % self.args.save_rate == 0 or epoch == self.args.n_epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path_model, f"transformer_{epoch}.pth"))
            # 2.5. 评估
            # self.evaluate(epoch)
        # 3. 清理工作
        self.writer.close()

    def evaluate(self):
        # 0. 开启推理模式
        print("----------------------Evaluate----------------------")
        self.model.eval()
        # 1. 加载测试数据（test.source.cn.txt）
        X, sources, targets = self.loader.load_test_data()
        # 2. 分批次测试
        for i in range(len(X) // self.args.batch_size_evaluate):
            start = i * self.args.batch_size_evaluate
            end = (i + 1) * self.args.batch_size_evaluate
            X_batch, sources_batch, targets_batch = X[start:end], sources[start:end], targets[start:end]

    def inference(self):
        # 0. 开启推理模式，加载模型参数
        print("----------------------Inference----------------------")
        self.model.eval()
        self.model.load_state_dict(torch.load(self.args.model_path))
        # 1. 加载测试数据
        X, sources, targets = self.loader.load_test_data()
        for i in range(self.args.n_inference):
            idx = np.random.randint(0, len(X))
            x = X[idx]
            x = torch.LongTensor(x).unsqueeze(0).to(self.args.device)
            y = torch.zeros(1, self.args.max_len, dtype=torch.int64, device=self.args.device)
            # 2. 测试
            predict = self.model(x, y)
            predict = torch.argmax(predict, dim=-1).detach().numpy()
            print(f"Question({i}) => {sources[i]}")
            print(f"Expected({i}) => {targets[i]}")
            print(f"Answer({i}) => {self.loader.embedding2text(predict.squeeze())}")
            print("------------------------------")

        
if __name__ == "__main__":

    # 1. 获取配置
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 初始化 Runner
    runner = Runner(args)

    # 3. 训练
    if args.inference:
        runner.inference()
    elif args.evaluate:
        runner.evaluate()
    else:
        runner.train()
