import os

from datetime import datetime

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.fusion_loss import FusionLoss
from net.network import Network
from config.config import DEVICE
from utils.data_loader import IVIFDataLoader, H5Dataset
from utils.early_stopping import EarlyStopping
from utils.utils import load_config


class NetTrain():
    def __init__(self, config):
        self.data_path = config['data_path']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.save_path = config['save_path']
        self.model_name = config['model_name']
        self.config = config

    def __call__(self, *args, **kwargs):
        train_loader = self.prepareData(self.data_path)
        model, optimizer, scheduler = self.buildModel()
        self.train(model, train_loader, optimizer, scheduler)

    def prepareData(self, data_path):
        train_data = H5Dataset(data_path, split='train', val_ratio=0.1, train_sample_ratio=1)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        tqdm.write(f"train samples: {len(train_data)}")
        return train_loader

    def buildModel(self):
        model = Network(self.config).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        return model, optimizer, scheduler

    def train(self, model, train_loader, optimizer, scheduler):
        model.train()
        fusion_loss = FusionLoss(config=self.config['loss'], batch_size=self.batch_size, epochs=self.epochs)

        # 从config读取早停参数
        es_mode = self.config.get('early_stop_mode', 'loss')
        es_patience = self.config.get('early_stop_patience', 10)
        es_eval_freq = self.config.get('early_stop_eval_freq', 5)

        early_stopping = EarlyStopping(
            patience=es_patience,
            mode=es_mode,
            eval_freq=es_eval_freq
        )

        start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        tqdm.write(f"train phase start, start time:{start_time}")
        tqdm.write(f"early stop mode: {es_mode}, patience: {es_patience}, eval_freq: {es_eval_freq}")

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            train_loader_tqdm = tqdm(train_loader)
            for ir, vis in train_loader_tqdm:
                ir_y = ir[:, 0:1]
                vis_y = vis[:, 0:1]
                output = model(ir_y, vis_y)
                loss = fusion_loss(output, ir, vis, start_time, epoch)

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                train_loader_tqdm.set_description(f"Epoch[{epoch}/{self.epochs}]")
                train_loader_tqdm.set_postfix_str(
                    f"total loss:{total_loss:.4f}, avg loss: {total_loss / len(train_loader):.4f}")

            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            tqdm.write(f"Epoch[{epoch}] train loss: {avg_loss:.6f}")

            if epoch > 20 and (epoch + 1) % 5 == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                torch.save(model.state_dict(), self.save_path + f"RTCI-Fuse_{epoch}.pth")

            should_stop = early_stopping(model, tqdm, current_loss=avg_loss)

            if should_stop:
                tqdm.write(f"Early stopping at epoch {epoch}")
                if early_stopping.best_model_state:
                    model.load_state_dict(early_stopping.best_model_state)
                break

        end_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        tqdm.write(f"train phase end, end time:{end_time}")

        self.saveModel(early_stopping.best_model_state or model.state_dict())

    def saveModel(self, model_state_dict):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = self.save_path + self.model_name
        torch.save(model_state_dict, model_path)


if __name__ == '__main__':
    config = load_config(f'./config.json')
    train_data_path = './h5/MSRS_imgsize_64.h5'
    epochs = 80
    lr = 1e-4
    batch_size = 16

    model_name = f'RTCI-Fuse.pth'
    config['data_path'] = train_data_path
    config['epochs'] = epochs
    config['batch_size'] = batch_size
    config['model_name'] = model_name
    config['save_path'] = f'./model/'
    config['lr'] = lr

    net = NetTrain(config)
    net()
