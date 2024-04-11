'''
Author       : wyx-hhhh
Date         : 2024-03-05
LastEditTime : 2024-04-11
Description  : 
'''
from trainers.base_train import BaseTrainer
from tqdm import tqdm


class GowallaTrainer(BaseTrainer):

    def __init__(self):
        super(GowallaTrainer, self).__init__()

    def train_model(self, model, train_loader, optimizer, device):
        model.train()

        pbar = tqdm(train_loader)
        epoch_loss = 0
        for data in pbar:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_loss += loss.item()
            pbar.set_description("Loss {}".format(round(epoch_loss, 4)))
        return epoch_loss

    def test_model(self, model, test_loader, device):
        pass
        # model.eval()
        # output = model(_,is_training=False)
        # user_embs = output['user_emb'].detach().cpu().numpy()
        # item_embs = output['item_emb'].detach().cpu().numpy()

        # test_user_list = list(test_gd.keys())

        # faiss_index = faiss.IndexFlatIP(hidden_size)
        # faiss_index.add(item_embs)

        # preds = dict()

        # for i in tqdm(range(0,len(test_user_list),1000)):
        #     user_ids = test_user_list[i:i+1000]
        #     batch_user_emb = user_embs[user_ids,:]
        #     D, I = faiss_index.search(batch_user_emb, 1000)

        #     for i, iid_list in enumerate(user_ids):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
        #         train_items = train_gd.get(user_ids[i],[])
        #         preds[user_ids[i]] = [x for x in list(I[i,:]) if x not in train_items]
        # return evaluate_recall(preds,test_gd, topN=topN)
