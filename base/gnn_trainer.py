'''
Author       : wyx-hhhh
Date         : 2024-04-11
LastEditTime : 2024-09-03
Description  : 
'''
import torch
from tqdm import tqdm
import faiss
from trainers.base_train import BaseTrainer
import time


class GnnTrainer(BaseTrainer):

    def __init__(self, config, evaluation_manager):
        super(GnnTrainer, self).__init__(config, evaluation_manager)

    def train_model(self, model, train_loader, optimizer, device, epoch):

        model.train()
        epoch_loss = 0

        start = time.time()
        for data in (pbar := tqdm(train_loader, desc="Training", unit="batch")):
            for key in data.keys():
                data[key] = data[key].to(device)
            output = model(data)
            loss = output['loss']
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.cpu().item()
            pbar.set_description(f"loss:  {loss.item():.4f}")
        res_dict = self.evaluation_manager.get_eval_res(
            loss=epoch_loss / len(train_loader),
            mode="train",
        )
        end = time.time()
        self.config["logger"].info(f"{self.config['model_name']} | {self.config['data']} | Train Metric | Epoch: {epoch}/{self.config['epoch']-1}")
        for key in res_dict.keys():
            self.config["logger"].info(f"{key}: {res_dict[key]}")
        self.config["logger"].info(f"The time of this epoch is {end - start}\n")
        return res_dict

    def test_model(self, model, train_gd, test_gd, hidden_size, epoch):
        model.eval()
        start = time.time()
        output = model(None, is_train=False)
        user_embs = output['user_embedding'].detach().cpu().numpy()
        item_embs = output['item_embedding'].detach().cpu().numpy()

        test_user_list = list(test_gd.keys())
        hidden_size = user_embs.shape[1]
        faiss_index = faiss.IndexFlatIP(hidden_size)  # 创建faiss索引
        faiss_index.add(item_embs)  # 添加item的embedding

        preds = dict()  # 存储预测结果

        for i in tqdm(range(0, len(test_user_list), 1000)):
            user_ids = test_user_list[i:i + 1000]  # 每次取1000个用户
            batch_user_emb = user_embs[user_ids, :]  # 取出这1000个用户的embedding
            D, I = faiss_index.search(batch_user_emb, 20)  # 检索最相似的n个item

            for i, uid_list in enumerate(user_ids):  # 遍历每个用户
                train_items = train_gd.get(user_ids[i], [])  # 获取用户的训练集
                preds[user_ids[i]] = [x for x in list(I[i, :]) if x not in train_items]  # 去除训练集中的item

        res_dict = self.evaluation_manager.get_eval_res(
            test_gd=test_gd,
            pred_gd=preds,
            mode="eval",
        )
        end = time.time()
        self.config["logger"].info(f"{self.config['model_name']} | {self.config['data']} | Test Metric | Epoch: {epoch}/{self.config['epoch']-1}")
        for key in res_dict.keys():
            self.config["logger"].info(f"{key}: {res_dict[key]}")
        self.config["logger"].info(f"The time of this epoch is {end - start}\n")
        return res_dict

    def test_model2(self, model, train_gd, test_gd, hidden_size, epoch):
        model.eval()
        start = time.time()

        pred_gd = {}
        with torch.no_grad():
            output = model(None, is_train=False)
            user_embs = output['user_embedding']
            item_embs = output['item_embedding']
            test_user_list = list(test_gd.keys())
            for i in range(0, len(test_user_list), 100):
                batch_user_list = test_user_list[i:i + 100]
                batch_user_list_gpu = torch.Tensor(batch_user_list).long().to(self.config["device"])

                # 计算当前批次用户的评分
                rating = model.get_users_rating(batch_user_list_gpu, user_embs, item_embs)

                # 处理排除的项目
                exclude_index = []
                exclude_items = []
                pos_item_dict = {user_id: train_gd.get(user_id, []) for user_id in batch_user_list}
                for range_i, items in enumerate(pos_item_dict.values()):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_index, exclude_items] = -(1 << 10)

                # 获取 top K 评分
                _, rating_K = torch.topk(rating, k=50)

                # 将结果保存到字典中
                for j, uid in enumerate(batch_user_list):
                    pred_gd[uid] = rating_K[j].tolist()
                del rating, rating_K, batch_user_list_gpu
                torch.cuda.empty_cache()  # 清理未使用的缓存

        res_dict = self.evaluation_manager.get_eval_res(
            test_gd=test_gd,
            pred_gd=pred_gd,
            mode="eval",
        )
        end = time.time()
        self.config["logger"].info(f"{self.config['model_name']} | {self.config['data']} | Test Metric | Epoch: {epoch}/{self.config['epoch']-1}")
        for key in res_dict.keys():
            self.config["logger"].info(f"{key}: {res_dict[key]}")
        self.config["logger"].info(f"The time of this epoch is {end - start}\n")
        return res_dict