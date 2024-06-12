import gradio as gr
import numpy as np

import base64


# 读取本地图片文件
def read_image():
    with open("/Users/wyx/程序/研究生/推荐系统/TorchCTR/data/CleanShot 2024-04-24 at 13.25.29@2x.png", "rb") as img_file:
        # 将图片文件转换为Base64编码
        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
        return encoded_img


encoded_img = read_image()
id_to_history_data = [
    [
        [2, "https://wyx-shanghai.oss-cn-shanghai.aliyuncs.com/img/20220307194854.png", "电子产品", 1000, 2],
    ],
    [
        [1, "https://wyx-shanghai.oss-cn-shanghai.aliyuncs.com/img/20220307194854.png", "图书", 20, 1],
        [2, "<img src='https://wyx-shanghai.oss-cn-shanghai.aliyuncs.com/img/20220204215500.png'>", "电子产品", 1000, 2],
        [3, "<img src='https://wyx-shanghai.oss-cn-shanghai.aliyuncs.com/img/20220307194854.png'>", "食品", 10, 3],
        [4, "<img src='https://wyx-shanghai.oss-cn-shanghai.aliyuncs.com/img/20220307194854.png'>", "家具", 500, 4],
        [5, f"<img src='data:image/png;base64,{encoded_img}'>", "服装", 200, 5],
    ],
]


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
            user_id = gr.Dropdown(choices=[0, 1], label="用户ID", value=1)
            # history_data = gr.Dataframe(id_to_history_data[user_id.value], label="历史行为数据", interactive=False)
            history_data = gr.Dataset(components=[gr.Textbox(visible=False), gr.HTML(visible=False)], label="历史行为数据", samples=id_to_history_data[user_id.value], headers=["用户ID", "用户头像"])
            # history_data = gr.Dataset(components=[gr.Textbox(visible=False), gr.Image(visible=False)], label="历史行为数据", samples=id_to_history_data[user_id.value], headers=["用户ID", "用户头像"])
            user_id.input(get_user_history_data, user_id, history_data)
        out = gr.DataFrame()

    btn = gr.Button("Run")
    btn.click(fn=next_step, inputs=[user_id], outputs=out)

app.launch()
