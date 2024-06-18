import pandas as pd
import gradio as gr
import numpy as np

import base64
from models.ddin.train_config import train_config

from utils.file_utils import get_file_path
from models.ddin.model import DDIN
from utils.utilities import get_values_by_keys

from managers import ConfigManager, DataManager, LoggerManager, TrainManager, SaveManager, EvaluationManager, save_manager


# 读取本地图片文件
def read_image(file_name):
    with open(get_file_path(['data', 'pixelrec', 'images', f"{file_name}.jpg"]), "rb") as img_file:
        # 将图片文件转换为Base64编码
        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
        encoded_html = f"<img src='data:image/png;base64,{encoded_img}'>"
        return encoded_html


pixel_inter_data = pd.read_csv(get_file_path(['data', 'pixelrec', 'Pixel200K_inter.csv']))
pixel_item_data = pd.read_csv(get_file_path(['data', 'pixelrec', 'Pixel200K_item.csv']))
# 读取json文件
pixel_map_dict = pd.read_json(get_file_path(['data', 'pixelrec', 'map_dict.json']), typ='series').to_dict()

user_map_dict = {v: k for k, v in pixel_map_dict['user_map_dict'].items()}
item_id_map_dict = {v: k for k, v in pixel_map_dict['item_id_map_dict'].items()}

config_manager = ConfigManager(train_config=train_config)
config = config_manager.get_config()
logger = LoggerManager(config=config)
data_manager = DataManager(config=config)
data_dict = data_manager.data_process()
train_df, test_df, test_dataloader, enc_dict = get_values_by_keys(data_dict, ['train_df', 'test_df', 'test_dataloader', 'enc_dict'])


# 从pixel_item_data中获取对应id的内容，只获取barrage_number,title,tag,description
def get_item_info(item_id):
    item_info = pixel_item_data[pixel_item_data['item_id'] == item_id][['title', 'tag', 'description']].values[0]
    return item_info


# 从train_df中随机获取5个用户的历史行为数据
id_to_history_data = {}
user_id_list = []
for i in range(5):
    user_id = train_df['user_id'].sample(1).values[0]
    user_history_data = train_df[(train_df['user_id'] == user_id) & (train_df['label'] == 1)]

    # 判断user_history_data的长度，如果长度小于5，则重新加载，如果长度大于5，则取前5个
    while len(user_history_data) < 3:
        user_id = train_df['user_id'].sample(1).values[0]
        user_history_data = train_df[(train_df['user_id'] == user_id) & (train_df['label'] == 1)]

    user_history_data = user_history_data.head(5)
    # 将user_history_data中的数据用user_map_dict和item_id_map_dict进行映射
    user_history_data['user_id'] = user_history_data['user_id'].map(user_map_dict)
    user_history_data['item_target_id'] = user_history_data['item_target_id'].map(item_id_map_dict)
    # 删除user_history_data中的item_target_tag，item_history_seq_id，item_history_seq_tag和label列
    user_history_data = user_history_data.drop(['item_target_tag', 'item_history_seq_id', 'item_history_seq_tag', 'label'], axis=1)
    new_user_history_data = []

    # 循环遍历user_history_data，获取每一行的数据
    for index, row in user_history_data.iterrows():
        # 获取item_target_id
        item_target_id = row['item_target_id']
        # 根据item_target_id获取item_target_id的数据
        item_target_info = get_item_info(item_target_id)
        # 将item_target_info和row进行合并
        new_user_history_data.append(pd.concat([pd.Series(item_target_info), row]))

    new_user_history_data = pd.DataFrame(new_user_history_data)
    new_user_history_data.columns = ['title', 'tag', 'description', 'user_id', 'item_target_id']
    user_id_list.append(user_id)
    new_user_history_data['image'] = new_user_history_data['item_target_id'].apply(read_image)
    new_user_history_data = new_user_history_data[['item_target_id', "image", 'title', 'tag', 'description']]
    id_to_history_data[user_id] = new_user_history_data.values.tolist()


def get_user_history_data(user_id):
    return id_to_history_data[user_id]


def next_step(user_id):
    recommend_data = gr.Dataframe(np.array([
        [1, "图书", 20, 1],
        [2, "电子产品", 1000, 2],
        [3, "食品", 10, 3],
    ]), label="推荐结果")
    return recommend_data


with gr.Blocks() as app:
    gr.Markdown("选择对应的用户id进行推荐")
    with gr.Row():

        with gr.Column():
            user_id = gr.Dropdown(label="用户ID", choices=user_id_list)
            history_data = gr.Dataset(components=[
                gr.Textbox(visible=False),
                gr.HTML(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
            ], label="历史行为数据", samples=id_to_history_data[user_id.value], headers=["物品ID", "物品图片", "物品标题", "物品标签", "物品描述"])
            # history_data = gr.Dataset(components=[gr.Textbox(visible=False), gr.Image(visible=False)], label="历史行为数据", samples=id_to_history_data[user_id.value], headers=["用户ID", "用户头像"])
            user_id.input(get_user_history_data, user_id, history_data)
        out = gr.DataFrame()

    btn = gr.Button("Run")
    btn.click(fn=next_step, inputs=[user_id], outputs=out)

app.launch()
