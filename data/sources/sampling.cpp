// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> negative_sampling(std::map<int, std::vector<int>> grouped_data, int neg_num, int item_num)
{
    // 每行包含用户ID、正样本和负样本数量
    int row = neg_num + 2;
    // 累计所有用户的正样本数量
    int total_pos_items = 0;

    // 计算所有用户的正样本总数
    for (const auto &pair : grouped_data)
    {
        total_pos_items += pair.second.size();
    }

    // 创建一个二维数组，行数为所有正样本数量，列数为 row
    py::array_t<int> S_array = py::array_t<int>({total_pos_items, row});
    py::buffer_info buf_S = S_array.request(); // 请求数组的缓冲区信息
    int *ptr = (int *)buf_S.ptr;               // 获取指向数组数据的指针

    int index = 0; // 用于跟踪当前填充的位置

    // 遍历每个用户
    for (const auto &pair : grouped_data)
    {
        int user = pair.first;
        const std::vector<int> &pos_item = pair.second;

        // 如果当前用户没有正样本，跳过该用户
        if (pos_item.empty())
        {
            continue;
        }

        // 遍历当前用户的每个正样本
        for (size_t i = 0; i < pos_item.size(); i++)
        {
            // 将用户ID存储在数组的第一列
            ptr[index * row] = user;
            // 将当前正样本存储在数组的第二列
            ptr[index * row + 1] = pos_item[i];

            // 生成负样本
            for (int neg_index = 0; neg_index < neg_num; neg_index++)
            {
                int negitem;
                do
                {
                    // 随机选择一个负样本（确保负样本不在正样本中）
                    negitem = rand() % item_num; // 假设 rand() 返回的负样本在 [0, item_num) 范围内
                } while (std::find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end()); // 确保负样本不在正样本中

                // 将生成的负样本存储在数组的后续列中
                ptr[index * row + neg_index + 2] = negitem;
            }
            index++; // 移动到下一行
        }
    }
    return S_array; // 返回包含用户、正样本和负样本的数组
}

py::array_t<int> negative_sampling_random(int user_num, int item_num, int train_num, std::map<int, std::vector<int>> grouped_data, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = grouped_data[user];

        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int negitem = 0;
            ptr[(user * perUserNum + pair_i) * row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for (int index = 2; index < neg_num + 2; index++)
            {
                do
                {
                    negitem = randint_(item_num);
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> negative_sampling_control(std::map<int, std::vector<int>> train_grouped,
                                           std::map<int, std::vector<int>> test_grouped,
                                           std::map<int, std::vector<int>> pred_grouped,
                                           int neg_num,
                                           int item_num,
                                           int error_item_ratio) // 新增参数：预测错误项目的复制次数
{
    int row = neg_num + 2; // 每行包含用户ID、正样本和负样本
    int total_pos_items = 0;

    for (const auto &pair : train_grouped)
    {
        total_pos_items += pair.second.size();
    }

    py::array_t<int> S_array = py::array_t<int>({total_pos_items, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    int index = 0; // 用于跟踪当前填充的位置

    for (const auto &pair : train_grouped)
    {
        int user = pair.first;
        const std::vector<int> &pos_items = pair.second;

        // 如果当前用户没有正样本，跳过该用户
        if (pos_items.empty())
        {
            continue;
        }

        // 每个用户单独生成 error_items
        std::vector<int> error_items; // 存储当前用户的预测错误项目

        // 收集当前用户的预测错误项目
        if (pred_grouped.find(user) != pred_grouped.end())
        {
            for (int item : pred_grouped[user])
            {
                if (find(pos_items.begin(), pos_items.end(), item) == pos_items.end() &&
                    find(test_grouped[user].begin(), test_grouped[user].end(), item) == test_grouped[user].end())
                {
                    // error_item_ratio次
                    for (int i = 0; i < error_item_ratio; i++)
                    {
                        error_items.push_back(item);
                    }
                }
            }
        }

        // 每个用户单独生成 all_neg_items
        std::vector<int> all_neg_items; // 存储当前用户的可用负样本

        // 收集当前用户的可用负样本
        for (int item = 0; item < item_num; item++)
        {
            if (find(pos_items.begin(), pos_items.end(), item) == pos_items.end())
            {
                all_neg_items.push_back(item);
            }
        }

        // 将all_neg_items个error_items集合到all_neg_items中
        all_neg_items.insert(all_neg_items.end(), error_items.begin(), error_items.end());
        // 利用all_neg_items生成负样本
        for (size_t i = 0; i < pos_items.size(); i++)
        {
            ptr[index * row] = user;
            ptr[index * row + 1] = pos_items[i];

            for (int neg_index = 0; neg_index < neg_num; neg_index++)
            {
                int negitem;
                do
                {
                    negitem = all_neg_items[rand() % all_neg_items.size()];
                } while (find(pos_items.begin(), pos_items.end(), negitem) != pos_items.end());

                ptr[index * row + neg_index + 2] = negitem;
            }
            index++;
        }
    }

    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

PYBIND11_MODULE(sampling, m)
{
    m.doc() = "negative sampling";
    m.def("set_seed", &set_seed, "set seed");
    m.def("negative_sampling", &negative_sampling, "negative sampling");
    m.def("negative_sampling_random", &negative_sampling_random, "negative sampling random");
    m.def("negative_sampling_control", &negative_sampling_control, "negative sampling control");
}

/*
<%
setup_pybind11(cfg)
%>
*/