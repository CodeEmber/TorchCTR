import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainers.base_train import BaseTrainer


class DeepLearningTrainer(BaseTrainer):

    def __init__(self, config, evaluation_manager):
        super(DeepLearningTrainer, self).__init__(config, evaluation_manager)

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        model.train()
        pred_list = []
        label_list = []
        loss_list = []
        user_list = []
        for data in train_loader:
            for key, value in data.items():
                data[key] = value.to(device)
            output_dict = model(data)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            pred_list.append(output_dict["y_pred"].detach().cpu().numpy())
            user_data = data.get(self.config["col_name"].get("user_col"), None)
            if user_data is not None:
                user_list.append(user_data.detach().cpu().numpy())
            label_list.append(data["label"].detach().cpu().numpy())
        if user_list:
            train_df = pd.DataFrame({
                self.config["col_name"].get("user_col"): np.concatenate(user_list).squeeze(),
                self.config["col_name"].get("pre_col"): np.concatenate(pred_list).squeeze(),
                self.config["col_name"].get("label_col"): np.concatenate(label_list).squeeze(),
            })
        else:
            train_df = None
        res_dict = self.evaluation_manager.get_eval_res(
            mode="train",
            y_true=label_list,
            y_pred=pred_list,
            test_df=train_df,
        )

        return res_dict

    def valid_model(
        self,
        model,
        valid_loader,
        device,
    ):
        model.eval()
        pred_list = []
        label_list = []
        loss_list = []
        user_list = []
        with torch.no_grad():
            for data in (pbar := tqdm(valid_loader)):
                for key, value in data.items():
                    data[key] = value.to(device)
                output_dict = model(data)
                loss = output_dict["loss"]
                loss_list.append(loss.item())
                user_data = data.get(self.config["col_name"].get("user_col"), None)
                if user_data is not None:
                    user_list.append(user_data.detach().cpu().numpy())
                pred_list.append(output_dict["y_pred"].detach().cpu().numpy())
                label_list.append(data["label"].detach().cpu().numpy())
                pbar.set_description(f"loss: {np.mean(loss_list):.4f}")
        if user_list:
            valid_df = pd.DataFrame({
                self.config["col_name"].get("user_col"): np.concatenate(user_list).squeeze(),
                self.config["col_name"].get("pre_col"): np.concatenate(pred_list).squeeze(),
                self.config["col_name"].get("label_col"): np.concatenate(label_list).squeeze(),
            })
        else:
            valid_df = None
        res_dict = self.evaluation_manager.get_eval_res(
            mode="valid",
            y_true=label_list,
            y_pred=pred_list,
            test_df=valid_df,
        )
        return res_dict

    def test_model(
        self,
        model,
        test_loader,
        device,
    ):
        model.eval()
        pred_list = []
        label_list = []
        loss_list = []
        user_list = []
        with torch.no_grad():
            for data in (pbar := tqdm(test_loader)):
                for key, value in data.items():
                    data[key] = value.to(device)
                output_dict = model(data)
                loss = output_dict["loss"]
                loss_list.append(loss.item())
                pred_list.append(output_dict["y_pred"].detach().cpu().numpy())
                user_data = data.get(self.config["col_name"].get("user_col"), None)
                if user_data is not None:
                    user_list.append(user_data.detach().cpu().numpy())
                label_list.append(data["label"].detach().cpu().numpy())
                pbar.set_description(f"loss: {np.mean(loss_list):.4f}")
        if user_list:
            test_df = pd.DataFrame({
                self.config["col_name"].get("user_col"): np.concatenate(user_list).squeeze(),
                self.config["col_name"].get("pre_col"): np.concatenate(pred_list).squeeze(),
                self.config["col_name"].get("label_col"): np.concatenate(label_list).squeeze(),
            })
        else:
            test_df = None
        res_dict = self.evaluation_manager.get_eval_res(
            mode="eval",
            y_true=label_list,
            y_pred=pred_list,
            test_df=test_df,
        )
        return res_dict

    def get_test_predict(self, model, test_loader, device):
        model.eval()
        pred_list = []

        for data in tqdm(test_loader):

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['y_pred']

            pred_list.extend(pred.squeeze().cpu().detach().numpy())

        return pred_list
