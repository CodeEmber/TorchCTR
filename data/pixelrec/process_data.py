import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from managers.logger_manager import logger
from data.base_process_data import BaseProcessData
from utils.file_utils import get_file_path
from data.pixelrec.dataset import get_pixelrec_dataset


class PixelRecProcessData(BaseProcessData):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def load_data(self):
        self.behaviour_df = pd.read_csv(get_file_path(self.config["behaviour_path"]))
        self.item_df = pd.read_csv(get_file_path(self.config["item_path"]))

        pos_df = pd.DataFrame()
        pos_df['user_id'] = self.behaviour_df['user_id']
        pos_df['item_id'] = self.behaviour_df['item_id']
        item_id2tag_map_dict = dict(zip(self.item_df['item_id'], self.item_df['tag']))
        pos_df['item_tag'] = pos_df['item_id'].map(item_id2tag_map_dict)

        user_map_dict = dict(zip(pos_df['user_id'].unique(), range(1, pos_df['user_id'].nunique() + 1)))
        item_id_map_dict = dict(zip(pos_df['item_id'].unique(), range(1, pos_df['item_id'].nunique() + 1)))
        item_tag_map_dict = dict(zip(pos_df['item_tag'].unique(), range(1, pos_df['item_tag'].nunique() + 1)))
        pos_df['user_id'] = pos_df['user_id'].map(user_map_dict).fillna(0).astype('int64')
        pos_df['item_id'] = pos_df['item_id'].map(item_id_map_dict).fillna(0).astype('int64')
        pos_df['item_tag'] = pos_df['item_tag'].map(item_tag_map_dict).fillna(0).astype('int64')

        pos_dict = pos_df.groupby('user_id')['item_id'].apply(list).to_dict()

        logger.info("数据加载完成")
        if self.config.get("debug", False):
            pos_df = pos_df.sample(frac=0.01).reset_index(drop=True)
            logger.info("调试模式，数据集缩小为原来的0.01")
        return pos_df, pos_dict

    def get_enc_dict(self, pos_df) -> dict:
        """本数据集对应的enc_dict

        Args:
            pos_df (DataFram): 处理后的数据集
        """
        self.enc_dict = {}
        self.enc_dict["user_id"] = {'vocab_size': pos_df['user_id'].nunique() + 1, 'type': 'user'}
        self.enc_dict["item_target_id"] = {'vocab_size': pos_df['item_id'].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_target_tag"] = {'vocab_size': pos_df['item_tag'].nunique() + 1, 'type': 'item'}
        self.enc_dict["item_history_seq_id"] = {'share_with': 'item_target_id', 'type': 'item'}
        self.enc_dict["item_history_seq_tag"] = {'share_with': 'item_target_tag', 'type': 'item'}
        self.enc_dict["count_map"] = {
            'user_id': pos_df['user_id'].value_counts().to_dict(),
            'item_target_id': pos_df['item_id'].value_counts().to_dict(),
            'item_target_tag': pos_df['item_tag'].value_counts().to_dict(),
        }

    def construct_data(self, pos_df, pos_dict, max_seq=20, neg_sample_ratio=2):
        label_list = []
        user_id_list = []
        item_target_id_list = []
        item_target_tag_list = []
        item_history_seq_id_list = []
        item_history_seq_tag_list = []
        all_user_list = pos_df['user_id'].unique()
        all_item_list, item_num = pos_df['item_id'].unique(), pos_df['item_id'].nunique()
        id2tag = dict(zip(pos_df['item_id'], pos_df['item_tag']))
        id2tag.update({0: 0})
        for user in tqdm(all_user_list, desc="构建数据集"):
            for i in range(len(pos_dict[user]) - 5, len(pos_dict[user]) - 1):
                user_id_list.append(user)
                item_target_id_list.append(pos_dict[user][i + 1])
                item_target_tag_list.append(id2tag[pos_dict[user][i + 1]])
                label_list.append(1)
                if i < max_seq:
                    item_history_seq_id_list.append(pos_dict[user][:i] + [0] * (max_seq - i))
                else:
                    item_history_seq_id_list.append(pos_dict[user][i - max_seq:i])
                item_history_seq_tag_list.append([id2tag[item] for item in item_history_seq_id_list[-1]])

                for i in range(neg_sample_ratio):
                    user_id_list.append(user)
                    label_list.append(0)
                    item_history_seq_id_list.append(item_history_seq_id_list[-1])
                    item_history_seq_tag_list.append(item_history_seq_tag_list[-1])

                    temp_item_index = random.randint(0, item_num - 1)
                    while all_item_list[temp_item_index] in pos_dict[user]:
                        temp_item_index = random.randint(0, item_num - 1)
                    item_target_id_list.append(all_item_list[temp_item_index])
                    item_target_tag_list.append(id2tag[all_item_list[temp_item_index]])
        data = {
            'user_id': user_id_list,
            'item_target_id': item_target_id_list,
            'item_target_tag': item_target_tag_list,
            'item_history_seq_id': item_history_seq_id_list,
            'item_history_seq_tag': item_history_seq_tag_list,
            'label': label_list,
        }
        logger.info("数据集构建完成,其中正样本数目为：{}，负样本数目为：{}".format(label_list.count(1), label_list.count(0)))
        data = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
        return data

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pos_df, pos_dict = self.load_data()
        self.get_enc_dict(pos_df)
        data = self.construct_data(pos_df, pos_dict, 20, self.config["neg_sample_ratio"])
        train_df = data[:int(len(data) * self.config["train_ratio"])].reset_index(drop=True)
        valid_df = data[int(len(data) * self.config["train_ratio"]):int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"]))].reset_index(drop=True)
        test_df = data[int(len(data) * (self.config["train_ratio"] + self.config["valid_ratio"])):].reset_index(drop=True)
        logger.info("数据集切分完成，其中训练集、验证集、测试集的比例为：{:.1f}:{:.1f}:{:.1f}".format(self.config["train_ratio"], self.config["valid_ratio"], 1 - self.config["train_ratio"] - self.config["valid_ratio"]))
        return train_df, valid_df, test_df

    def get_dataset(self, data: pd.DataFrame, enc_dict: dict = None) -> Dataset:
        dataset = get_pixelrec_dataset(data, self.config, self.enc_dict)
        return dataset
