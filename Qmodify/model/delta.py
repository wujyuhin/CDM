# Load the data from files
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA


class Delta():
    def __init__(self, q_m, R, stu_num, prob_num, know_num,epsilon=0.05):
        self.q_m = q_m
        self.R = R
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.modify_q_m = q_m.copy()
        self.epsilon = 0.05

    # ========================= 计算 delta ====================================
    def delta(self, CDM, item):
        """ 计算第j道题目的delta
        :param CDM: 经过EduCDM中的DINA训练的cdm模型实例
        :param item: 第j道题目
        :return:  返回第j道题目的delta
        """
        # item:第j道题目
        # cdm模型对象
        g = CDM.guess[item]
        s = CDM.slip[item]
        return 1 - g - s

    # ====================== 掌握n种属性的q向量的所有情况 ==========================
    def generate_q_state(self, nums: list):
        """
        生成所有q向量的情况，例如如果Q矩阵是4维，输入的nums=[1,3]，
        则生成的q向量是[0,1,0,1]的所有情况有[1,0,0,1]、[0,1,1,1]
        :param nums: 已经确定的属性如已确定第1、3个属性为1，则nums=[1,3]
        :return: 返回所有q向量的情况：nparray([[1,0,0,1],[0,1,1,1]])
        """
        nums_list = nums.copy()
        # q_m Q矩阵
        # nums是指定列为全1的索引 nums = [1,3,5]
        if len(nums_list) == 0:
            return np.eye(self.know_num)
        else:
            e = np.eye(self.know_num - len(nums_list))
            insert_one = np.ones(self.know_num - len(nums_list))  # 已经确定3种情况下，插入1向量长度应该是8-3
            nums_list.sort()
            for num in nums_list:
                e = np.insert(e, num, insert_one, axis=1)
            return e

    # =================================== 遍历所有的q向量情况求最大的delta  ================================

    def cal_dalta(self, item, one_q):
        """
        计算替换了q向量后的第j道题目的delta
        :param item: 第j道题目
        :param one_q: 是定修改的q向量
        :return:  返回delta
        """
        # q_m Q矩阵
        # 第j道题目
        # one_q 是定修改的q向量
        # 功能：Q矩阵的第j行修改为one_q
        modify_q_m = self.q_m.copy()
        modify_q_m[item, :] = one_q
        cdm = DINA(self.R, modify_q_m, self.stu_num, self.prob_num, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        return self.delta(cdm, item)

    # ===========================================  顺序搜索满足delta最大的q向量(利用q(11)-q(1)>ε)  =========================
    # 遍历掌握1种属性的所有q向量情况下最大的delta
    def cal_max_delta(self, item, q_list):
        """
        遍历q_list中的所有q情况下最大的delta
        :param q_list: [[1,0,0],[0,1,0],[0,0,1]]
        :param item:  第j道题目
        :return:    返回最大的delta与对应的行索引
        """
        delta_list = []
        for i in range(q_list.shape[0]):
            delta_list.append(self.cal_dalta(item, q_list[i, :]))
        max_value = max(delta_list)
        max_index = delta_list.index(max_value)
        return max_value, max_index

    def modify_qvector(self, item, epsilon):
        """
        修改第题目item的q向量，且满足q(11)-q(1)>ε
        :param item:  第j道题目
        :param epsilon:  误差
        :return:
        """
        # ================  第一次迭代  ===========================================================
        nums = []
        q_list = self.generate_q_state(nums)
        max_value0, max_row_index0 = self.cal_max_delta(item, q_list)
        self.modify_q_m[item, :] = q_list[max_row_index0, :]
        # =================  第二次迭代(以确定一列为1)   ============================================
        nums.append(max_row_index0)
        q_list = self.generate_q_state(nums)
        max_value1, max_row_index1 = self.cal_max_delta(item, q_list)
        for num in nums:
            q_list[max_row_index1, num] = 0
        max_col_index = np.argmax(q_list[max_row_index1, :])
        new_delta, old_delta = max_value1, max_value0
        while new_delta - old_delta > epsilon:
            self.modify_q_m[item, max_col_index] = 1
            nums.append(max_col_index)
            q_list = self.generate_q_state(nums)
            max_value, max_row_index = self.cal_max_delta(item, q_list)
            for num in nums:
                q_list[max_row_index, num] = 0
            max_col_index = np.argmax(q_list[max_row_index, :])
            old_delta = new_delta
            new_delta = max_value
        return self.modify_q_m

    def modify_Q(self):
        for item in tqdm(range(self.prob_num)):
            self.modify_q_m = self.modify_qvector(item, self.epsilon)
        return self.modify_q_m


# ========================= 计算 delta ====================================
# def delta(cdm, item):
#     # item:第j道题目
#     # cdm模型对象
#     g = cdm.guess[item]
#     s = cdm.slip[item]
#     return 1 - g - s


# ====================== 掌握n种属性的q向量的所有情况 ==========================
# def generate_q_state(q_m, nums: list):
#     nums_list = nums.copy()
#     # q_m Q矩阵
#     # nums是指定列为全1的索引 nums = [1,3,5]
#     num_q = q_m.shape[1]
#     if len(nums_list) == 0:
#         return np.eye(num_q)
#     else:
#         e = np.eye(num_q - len(nums_list))
#         insert_one = np.ones(num_q - len(nums_list))  # 已经确定3种情况下，插入1向量长度应该是8-3
#         nums_list.sort()
#         for num in nums_list:
#             e = np.insert(e, num, insert_one, axis=1)
#         return e


# =================================== 遍历所有的q向量情况求最大的delta  ================================

# def cal_dalta(q_m, item, one_q):
#     # q_m Q矩阵
#     # 第j道题目
#     # one_q 是定修改的q向量
#     # 功能：Q矩阵的第j行修改为one_q
#     modify_q_m = q_m.copy()
#     modify_q_m[item, :] = one_q
#     cdm = DINA(R, modify_q_m, stu_num, prob_num, know_num, skip_value=-1)
#     cdm.train(epoch=2, epsilon=1e-3)
#     return delta(cdm, item)


# one_q = generate_q_state(q_m, [1, 2])[1, :]
# cal_dalta(q_m,4,one_q)

# ===========================================  顺序搜索满足delta最大的q向量(利用q(11)-q(1)>ε)  =========================

# 遍历掌握1种属性的所有q向量情况下最大的delta
# def cal_max_delta(q_list, item):
#     delta_list = []
#     for i in range(q_list.shape[0]):
#         delta_list.append(cal_dalta(q_m, item, q_list[i, :]))
#     max_value = max(delta_list)
#     max_index = delta_list.index(max_value)
#     return max_value, max_index


# def modify_qvector(q_m, item, epsilon=0):
#     # ================  第一次迭代  ===========================================================
#     nums = []  # 已经确定的属性
#     q_list = generate_q_state(q_m, nums)  # 单位阵
#     max_value0, max_row_index0 = cal_max_delta(q_list, item)  # 计算最大的delta与对应的行索引
#     modify_q_m[item, :] = q_list[max_row_index0, :]  # 修改Q矩阵
#     # =================  第二次迭代(以确定一列为1)   ============================================
#     nums.append(max_row_index0)
#     q_list = generate_q_state(modify_q_m, nums)  # 继续生成具有两个知识点的q所有种类
#     max_value1, max_row_index1 = cal_max_delta(q_list, item)  # 计算最大的delta
#     # 根据最大的delta寻找可能要修改的属性
#     for num in nums:
#         q_list[max_row_index1, num] = 0
#     max_col_index = np.argmax(q_list[max_row_index1, :])  # 需要固定属性为1的索引
#     # 重命名
#     new_delta, old_delta = max_value1, max_value0
#     # 进入循环判断是否需要第三次迭代
#     while new_delta - old_delta > epsilon:
#         modify_q_m[item, max_col_index] = 1  # 修改Q矩阵
#         nums.append(max_col_index)
#         q_list = generate_q_state(modify_q_m, nums)  # 生成q所有种类
#         max_value, max_row_index = cal_max_delta(q_list, item)  # 计算最大delta
#         for num in nums:
#             q_list[max_row_index, num] = 0
#         max_col_index = np.argmax(q_list[max_row_index, :])  # 计算最大delta的索引
#         old_delta = new_delta
#         new_delta = max_value
#     return modify_q_m

# if __name__ == '__main__':
#     q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')
#     modify_q_m = q_m.copy()
#     prob_num, know_num = q_m.shape[0], q_m.shape[1]
#     R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))
#     stu_num = R.shape[0]
#
#     # ============================ 第一步 估计gs ==================================
#
#     cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
#
#     cdm.train(epoch=2, epsilon=1e-3)
#     cdm.save("dina.params")
#     for item in tqdm(range(modify_q_m.shape[0])):
#         modify_q_m = modify_qvector(modify_q_m, item, epsilon=0)

if __name__ == '__main__':
    q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')
    prob_num, know_num = q_m.shape[0], q_m.shape[1]
    R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))
    stu_num = R.shape[0]
    # ================= delta法 ==================================
    delta = Delta(q_m, R, stu_num, prob_num, know_num)
    delta_model = Delta(q_m, R, stu_num, prob_num, know_num)
    modify_q_m1 = delta_model.modify_qvector(item=0, epsilon=0.05)  # 修改第0道题目的q向量
    modify_q_m2 = delta_model.modify_Q(epislon=0.05)  # 修改所有题目的q向量
