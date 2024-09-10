'''
Author       : wyx-hhhh
Date         : 2024-09-04
LastEditTime : 2024-09-10
Description  : 
'''
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from base.recommender import Recommender


class GraphRecommender(Recommender):

    def __init__(self, train_config):
        super(GraphRecommender, self).__init__(train_config)
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.epoch = 0

    def train(self):
        self.model.train()
        self.data_dict["train_dataloader"].dataset.reset_negative_sampling()
        train_loader = self.data_dict["train_dataloader"]
        epoch_loss = 0

        start = time.time()
        for data in (pbar := tqdm(train_loader, desc="Training", unit="batch")):
            for key in data.keys():
                data[key] = data[key].to(self.config["device"])
            output = self.model(data)
            loss = output['loss']
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.cpu().item()
            pbar.set_description(f"loss:  {loss.item():.4f}")
        res_dict = self.evaluation_manager.get_eval_res(
            loss=epoch_loss / len(train_loader),
            mode="train",
        )
        self.epoch += 1
        end = time.time()
        self.config["logger"].info(f"{self.config['model_name']} | {self.config['data']} | Train Metric | Epoch: {self.epoch}/{self.config['epoch']-1}")
        for key in res_dict.keys():
            self.config["logger"].info(f"{key}: {res_dict[key]}")
        self.config["logger"].info(f"The time of this epoch is {end - start}\n")
        return res_dict

    def get_users_rating(self, user_ids, user_embs, item_embs):
        user_embs = user_embs[user_ids.long()]
        rating = torch.matmul(user_embs, item_embs.t())
        return rating

    def test(self):
        test_gd = self.data_dict["test_grouped_data"]
        train_gd = self.data_dict["train_grouped_data"]
        self.model.eval()
        start = time.time()

        self.pred_gd = {}
        max_k = max(k for eval_metric in self.config["metric_func"]["eval"] for k in eval_metric["k"])
        with torch.no_grad():
            output = self.model(None, is_train=False)
            user_embs = output['user_embedding']
            item_embs = output['item_embedding']
            test_user_list = list(test_gd.keys())
            for i in range(0, len(test_user_list), 100):
                batch_user_list = test_user_list[i:i + 100]
                batch_user_list_gpu = torch.Tensor(batch_user_list).long().to(self.config["device"])

                rating = self.get_users_rating(batch_user_list_gpu, user_embs, item_embs)

                exclude_index = []
                exclude_items = []
                pos_item_dict = {user_id: train_gd.get(user_id, []) for user_id in batch_user_list}
                for range_i, items in enumerate(pos_item_dict.values()):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_index, exclude_items] = -(1 << 10)

                _, rating_K = torch.topk(rating, k=max_k)

                for j, uid in enumerate(batch_user_list):
                    self.pred_gd[uid] = rating_K[j].tolist()
                del rating, rating_K, batch_user_list_gpu
                torch.cuda.empty_cache()
        self.config["data_dict"]["pred_gd"] = self.pred_gd
        res_dict = self.evaluation_manager.get_eval_res(
            test_gd=test_gd,
            pred_gd=self.pred_gd,
            mode="eval",
        )
        end = time.time()
        self.config["logger"].info(f"{self.config['model_name']} | {self.config['data']} | Test Metric | Epoch: {self.epoch}/{self.config['epoch']-1}")
        for key in res_dict.keys():
            self.config["logger"].info(f"{key}: {res_dict[key]}")
        self.config["logger"].info(f"The time of this epoch is {end - start}\n")
        return res_dict

    def run(self):
        for i in range(self.config['epoch']):
            if self.config.get("early_stop", False) and self.config.get("is_early_stopping", False):
                self.config["logger"].info(f"Best metric: {self.save_manager.early_stopping.best_metric}")
                break

            train_metric = self.train()
            test_metric = self.test()
            self.save_manager.save_all(
                epoch=i,
                train_metric=train_metric,
                valid_metric=None,
                test_metric=test_metric,
                other_metric=None,
                model=self.model,
            )
        self.save_manager.save_run_info(self.config)
