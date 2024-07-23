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

py::array_t<int> negative_sampling(int user_num, int item_num, int train_num, std::map<int, std::vector<int>> grouped_data)
{
    // 创建一个一维数组，用于存储负样本
    vector<int> neg_samples;
    // 根据上面的变量，计算出负样本
    for (int i = 0; i < user_num; i++)
    {
        // 用于存储用户已经交互过的物品
        vector<int> pos_items = grouped_data[i];
        int user_pos_num = pos_items.size();
        // 根据用户已经交互过的物品，计算出对应长度的负样本
        for (int j = 0; j < user_pos_num; j++)
        {
            // 每一个正样本随机生成1个负样本，并且存储到neg_samples中
            for (int k = 0; k < 1; k++)
            {
                int neg_item = randint_(item_num);
                while (find(pos_items.begin(), pos_items.end(), neg_item) != pos_items.end())
                {
                    neg_item = randint_(item_num);
                }
                neg_samples.push_back(neg_item);
            }
        }
    }
    // 将负样本转换为numpy数组
    py::array_t<int> neg_samples_np = py::array_t<int>(neg_samples.size(), neg_samples.data());
    return neg_samples_np;
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
}

/*
<%
setup_pybind11(cfg)
%>
*/