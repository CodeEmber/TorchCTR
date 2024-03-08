'''
Author       : wyx-hhhh
Date         : 2024-03-05
LastEditTime : 2024-03-08
Description  : 
'''
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_func: List[str] = ["roc_auc_score", "log_loss"],
):
    model.train()
    pred_list = []
    label_list = []
    loss_list = []
    for data in (pbar := tqdm(train_loader)):
        for key, value in data.items():
            data[key] = value.to(device)
        output_dict = model(data)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
        pred_list.append(output_dict["y_pred"].detach().cpu().numpy())
        label_list.append(data["label"].detach().cpu().numpy())
        pbar.set_description(f"loss: {np.mean(loss_list):.4f}")
    res_dict = dict()
    for metric in metric_func:
        try:
            if metric == "roc_auc_score":
                res_dict[metric] = roc_auc_score(np.concatenate(label_list), np.concatenate(pred_list))
            elif metric == "log_loss":
                res_dict[metric] = log_loss(np.concatenate(label_list), np.concatenate(pred_list), eps=1e-7)
            else:
                res_dict[metric] = eval(metric)(np.concatenate(label_list), np.concatenate(pred_list))
        except:
            raise ValueError(f"metric_func must be roc_auc_score, log_loss or {metric}")
    return res_dict


def test_model(model, test_loader, device, metric_func=["roc_auc_score", "log_loss"]):
    model.eval()
    pred_list = []
    label_list = []
    loss_list = []
    with torch.no_grad():
        for data in (pbar := tqdm(test_loader)):
            for key, value in data.items():
                data[key] = value.to(device)
            output_dict = model(data)
            loss = output_dict["loss"]
            loss_list.append(loss.item())
            pred_list.append(output_dict["y_pred"].detach().cpu().numpy())
            label_list.append(data["label"].detach().cpu().numpy())
            pbar.set_description(f"loss: {np.mean(loss_list):.4f}")
    res_dict = dict()
    for metric in metric_func:
        try:
            if metric == "roc_auc_score":
                res_dict[metric] = roc_auc_score(np.concatenate(label_list), np.concatenate(pred_list))
            elif metric == "log_loss":
                res_dict[metric] = log_loss(np.concatenate(label_list), np.concatenate(pred_list), eps=1e-7)
            else:
                res_dict[metric] = eval(metric)(np.concatenate(label_list), np.concatenate(pred_list))
        except:
            raise ValueError(f"metric_func must be roc_auc_score, log_loss or {metric}")
    return res_dict


def get_test_predict(model, test_loader, device):
    model.eval()
    pred_list = []

    for data in tqdm(test_loader):

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['y_pred']

        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return pred_list
