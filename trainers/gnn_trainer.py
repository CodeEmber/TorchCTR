'''
Author       : wyx-hhhh
Date         : 2024-04-11
LastEditTime : 2024-07-03
Description  : 
'''
import pandas as pd
from tqdm import tqdm
import faiss
from trainers.base_train import BaseTrainer
from utils.evaluation import evaluate_recall


class GraphNeuralNetworkTrainer(BaseTrainer):

    def __init__(self, config, evaluation_manager):
        super(GraphNeuralNetworkTrainer, self).__init__(config, evaluation_manager)

    def train_model(self, model, train_loader, optimizer, device):
        model.train()

        epoch_loss = 0
        for data in train_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_loss += loss.item()
        res_dict = self.evaluation_manager.get_eval_res(
            loss=epoch_loss / len(train_loader),
            mode="train",
        )
        return res_dict

    def test_model(self, model, train_gd, test_gd, hidden_size):
        model.eval()

        output = model(None, is_train=False)
        user_embs = output['user_embedding'].detach().cpu().numpy()
        item_embs = output['item_embedding'].detach().cpu().numpy()

        test_user_list = list(test_gd.keys())

        faiss_index = faiss.IndexFlatIP(hidden_size)  # 创建faiss索引
        faiss_index.add(item_embs)  # 添加item的embedding

        preds = dict()  # 存储预测结果

        for i in tqdm(range(0, len(test_user_list), 1000)):
            user_ids = test_user_list[i:i + 1000]  # 每次取1000个用户
            batch_user_emb = user_embs[user_ids, :]  # 取出这1000个用户的embedding
            D, I = faiss_index.search(batch_user_emb, 100)  # 检索最相似的100个item

            for i, uid_list in enumerate(user_ids):  # 遍历每个用户
                train_items = train_gd.get(user_ids[i], [])  # 获取用户的训练集
                preds[user_ids[i]] = [x for x in list(I[i, :]) if x not in train_items]  # 去除训练集中的item
        # 将preds构建成test_df的格式
        test_df = []
        for user in preds:
            for i, item in enumerate(preds[user]):
                if item in test_gd[user]:
                    test_df.append([user, item, i + 1, 1])
                else:
                    test_df.append([user, item, i + 1, 0])
        test_df = pd.DataFrame(test_df, columns=['user_id', 'item_id', 'ranking', 'label'])
        res_dict = self.evaluation_manager.get_eval_res(
            test_df=test_df,
            mode="eval",
        )
        return res_dict
