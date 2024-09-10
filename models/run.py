# '''
# Author       : wyx-hhhh
# Date         : 2024-08-23
# LastEditTime : 2024-09-03
# Description  :
# '''
# import multiprocessing
# import torch
# from managers import ConfigManager, DataManager, LoggerManager, SaveManager, EvaluationManager
# from trainers.gnn_trainer import GnnTrainer
# from utils.torch_utils import set_device
# from utils.utilities import set_seed

# def gnn_run(model, config):
#     data_dict = DataManager(config=config).data_process()
#     evaluation_manager = EvaluationManager(config=config)
#     train_manager: GnnTrainer = TrainManager(config=config, evaluation_manager=evaluation_manager).trainer
#     save_manager = SaveManager(config=config)

#     device = set_device(config['device'])
#     graph_data = data_dict["graph_data"].to(device)
#     model = model(config=config, g=graph_data)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
#     model = model.to(device)
#     config["logger"].send_message(config, message_type=0, message_content_type=0)
#     for i in range(config['epoch']):
#         if config.get("early_stop", False) and config.get("is_early_stopping", False):
#             config["logger"].info(f"Best metric: {save_manager.early_stopping.best_metric}")
#             break
#         if config["model_name"] == "sgl":
#             model.generation_augmented_graph()
#         if config["model_name"] == "ncl":
#             model.e_step()
#         data_dict["train_dataloader"].dataset.reset_negative_sampling()
#         train_metric = train_manager.train_model(
#             model=model,
#             train_loader=data_dict["train_dataloader"],
#             optimizer=optimizer,
#             device=device,
#             epoch=i,
#         )

#         #模型验证
#         test_metric = None
#         if i % config.get("eval_step", 1) == 0 or i == config['epoch'] - 1:
#             test_metric = train_manager.test_model2(
#                 model=model,
#                 train_gd=data_dict["train_grouped_data"],
#                 test_gd=data_dict["test_grouped_data"],
#                 hidden_size=config['embedding_dim'],
#                 epoch=i,
#             )

#         save_manager.save_all(
#             epoch=i,
#             train_metric=train_metric,
#             valid_metric=None,
#             test_metric=test_metric,
#             other_metric=None,
#             model=model,
#         )
#     save_manager.save_run_info(config)

# def run(model, train_config):
#     # 获取训练配置
#     config = ConfigManager(train_config=train_config).get_config()
#     logger = LoggerManager(config=config)
#     config['logger'] = logger
#     set_seed(config.get("seed", 2024))

#     # 训练模型
#     if config['trainer'] == 'gnn':
#         gnn_run(model, config)
