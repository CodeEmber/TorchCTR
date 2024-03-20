# TorchCTR

![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg) ![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-blue.svg) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

TorchCTR是使用Pytorch编写的点击率预测开源库，旨在帮助研究者快速复现和实现各种推荐算法模型提供参考。

本代码库是个人学习过程中的编写的代码，代码没有经过严格测试，如果发现bug，欢迎PR。

---

## 👋 Installation

使用conda进行虚拟环境管理，可以使用下面的命令安装相关python库。

```bash
conda env update -f=environment.yaml
```

---

## 🥳 Getting Started

1. 从每个模型的run_expid.py文件进入

   ```bash
   python -m models.deepfm.run_expid.py
   ```
2. 从main.py文件进入

   ```bash
   python main.py
   ```
3. 使用tensorboard查看训练过程和运行结果

   ```bash
   tensorboard --logdir {results_path} --load_fast=false
   ```

---

## 📊 Models

|    Model    |                                                  Paper                                                  |
| :---------: | :------------------------------------------------------------------------------------------------------: |
| Wide & Deep |                               Wide & Deep Learning for Recommender Systems                               |
|   DeepFM   |                 DeepFM: A Factorization-Machine based Neural Network for CTR Prediction                 |
|     NFM     |                      Neural Factorization Machines for Sparse Predictive Analytics                      |
|   FiBiNET   | FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction |
|     NCF     |                                     Neural Collaborative Filtering∗                                     |

---

## 📝 Todo List

- [ ] 添加更多数据集
  - [ ] 添加Frappe数据集
  - [X] 添加amazon数据集
  - [X] 添加MovieLens数据集
  - [ ] 添加Avazu数据集
  - [ ] 添加KDD12数据集
- [ ] 修改main文件，使其支持命令行输入
- [ ] 接入slack，方便查看程序运行情况

---

## 📚 Reference

推荐系统入门的书籍推荐王喆的《深度学习推荐系统》，书中详细介绍了推荐系统的基本知识和常用模型，同时也介绍了一些工业界的实践经验，记录的笔记可以参考下面的链接。
还有一个链接是个人在阅读论文过程中的一些总结，主要是对论文的一些重点和难点进行了总结，希望能够帮助到大家。

- [《深度学习推荐系统》](https://weread.qq.com/web/bookDetail/b7732f20813ab7c33g015dea)
- [推荐系统学习笔记](https://www.wolai.com/wyx-hhhh/9AzgMp2jcfaVkdZusY1Biv)
- [推荐系统论文阅读笔记](https://www.wolai.com/wyx-hhhh/b47zdaJfje2eqv5w39JstG)
