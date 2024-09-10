'''
Author       : wyx-hhhh
Date         : 2024-09-04
LastEditTime : 2024-09-05
Description  : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        """计算BPR损失

        Args:
            user_emb (torch.Tensor): 用户embedding
            pos_emb (torch.Tensor): 正样本embedding
            neg_emb (torch.Tensor): 负样本embedding

        Returns:
            torch.Tensor: 损失
        """
        pos_score = torch.sum(user_emb * pos_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_emb, dim=1)
        # loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))
        return loss


class L2RegLoss(nn.Module):

    def __init__(self, reg: float, batch_size: int):
        super(L2RegLoss, self).__init__()
        self.reg = reg
        self.batch_size = batch_size

    def forward(self, *args) -> torch.Tensor:
        """计算L2正则化损失

        Args:
            *args: 需要计算正则化的embedding

        Returns:
            torch.Tensor: L2正则化损失
        """
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)**2
        emb_loss = emb_loss / 2
        return emb_loss * self.reg / self.batch_size


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """计算InfoNCE损失

        Args:
            view1 (torch.Tensor): 第一个视图的嵌入
            view2 (torch.Tensor): 第二个视图的嵌入

        Returns:
            torch.Tensor: InfoNCE损失
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
