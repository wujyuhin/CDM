# Load the data from files
import numpy as np
import json
import pandas as pd

q_m = np.loadtxt("../../../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')
modify_q_m = q_m.copy()
prob_num, know_num = q_m.shape[0], q_m.shape[1]

# training data
#with open("../../../data/math2015/FrcSub/simulation_data.json", encoding='utf-8') as file:
    #train_set = json.load(file)
#stu_num = max([x['user_id'] for x in train_set]) + 1
#R = -1 * np.ones(shape=(stu_num, prob_num))
#for log in train_set:
    #R_init = R[log['user_id'], log['item_id']]
    #s = log['score']
    #R[log['user_id'], log['item_id']] = log['score']

R = np.array(pd.read_csv("../../../data/math2015/FrcSub/加减法data.csv", index_col=0))
stu_num=R.shape[0]
# testing data
#with open("../../../data/math2015/FrcSub/simulation_data.json", encoding='utf-8') as file:
    #test_set = json.load(file)

# ============================ 第一步 估计gs ==================================
import logging

logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA

cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)

cdm.train(epoch=2, epsilon=1e-3)
cdm.save("dina.params")


# ========================= 计算 delta ====================================
def delta(cdm, item):
    # item:第j道题目
    # cdm模型对象
    g = cdm.guess[item]
    s = cdm.slip[item]
    return 1 - g - s


# ====================== 掌握n种属性的q向量的所有情况 ==========================
def generate_q_state(q_m, nums: list):
    nums_list = nums.copy()
    # q_m Q矩阵
    # nums是指定列为全1的索引 nums = [1,3,5]
    num_q = q_m.shape[1]
    if len(nums_list) == 0:
        return np.eye(num_q)
    else:
        e = np.eye(num_q - len(nums_list))
        insert_one = np.ones(num_q - len(nums_list))  # 已经确定3种情况下，插入1向量长度应该是8-3
        nums_list.sort()
        for num in nums_list:
            e = np.insert(e, num, insert_one, axis=1)
        return e


# =================================== 遍历所有的q向量情况求最大的delta  ================================

def cal_dalta(q_m, item, one_q):
    # q_m Q矩阵
    # 第j道题目
    # one_q 是定修改的q向量
    # 功能：Q矩阵的第j行修改为one_q
    modify_q_m = q_m.copy()
    modify_q_m[item, :] = one_q
    cdm = DINA(R, modify_q_m, stu_num, prob_num, know_num, skip_value=-1)
    cdm.train(epoch=2, epsilon=1e-3)
    return delta(cdm, item)


# one_q = generate_q_state(q_m, [1, 2])[1, :]
# cal_dalta(q_m,4,one_q)

# ===========================================  顺序搜索满足delta最大的q向量(利用q(11)-q(1)>ε)  =========================

# 遍历掌握1种属性的所有q向量情况下最大的delta
def cal_max_delta(q_list, item):
    delta_list = []
    for i in range(q_list.shape[0]):
        delta_list.append(cal_dalta(q_m, item, q_list[i, :]))
    max_value = max(delta_list)
    max_index = delta_list.index(max_value)
    return max_value, max_index


def modify_qvector(q_m, item, epsilon=0):
    # 第一次迭代
    nums = []
    q_list = generate_q_state(q_m, nums)
    max_value0, max_row_index0 = cal_max_delta(q_list, item)  # 计算最大的delta与对应的行索引
    modify_q_m[item, :] = q_list[max_row_index0, :]  # 修改Q矩阵
    # 第二次迭代(以确定一列为1)
    nums.append(max_row_index0)
    q_list = generate_q_state(modify_q_m, nums)
    # 修改q矩阵
    max_value1, max_row_index1 = cal_max_delta(q_list, item)  # 计算最大的delta
    for num in nums:
        q_list[max_row_index1, num] = 0
    max_col_index = np.argmax(q_list[max_row_index1, :])  # 需要固定属性为1的索引
    # 重命名
    new_delta, old_delta = max_value1, max_value0
    while new_delta - old_delta > epsilon:
        modify_q_m[item, max_col_index] = 1  # 修改Q矩阵
        nums.append(max_col_index)
        q_list = generate_q_state(modify_q_m, nums)  # 生成q所有种类
        max_value, max_row_index = cal_max_delta(q_list, item)  # 计算最大delta
        for num in nums:
            q_list[max_row_index, num] = 0
        max_col_index = np.argmax(q_list[max_row_index, :])  # 计算最大delta的索引
        old_delta = new_delta
        new_delta = max_value
    return modify_q_m


for item in range(modify_q_m.shape[0]):
    modify_q_m = modify_qvector(modify_q_m, item, epsilon=0)
# modify_q_m = modify_qvector(modify_q_m, item=20, epsilon=0.05)
# flag = 2
# while max_value1 - max_value0 > epsilon:
#     nums.append(max_col_index)
#     q_list = generate_q_state(modify_q_m, nums)
#     max_value, max_row_index = cal_max_delta(q_list)
#     for num in nums:
#         q_list[max_row_index, num] = 0
#     max_col_index = np.argmax(q_list[max_row_index, :])
#     modify_q_m[item, max_col_index] = 1  # 修改Q矩阵
#     # 更新不同迭代次数的delta差值
#     max_value0 = max_value1
#     max_value1 = max_value
#     print('迭代了一次')
#     flag += 1
#     if flag == 8:
#         break

# ======================== 全部q状态进行遍历求delta  ======================================
# item = 5
# modify2_q_m = q_m.copy()
# deltas = []
# for row in range(1, cdm.all_states.shape[0]):
#     one_q = cdm.all_states[row]
#     deltas.append(cal_dalta(q_m, item, one_q))
#     print(row)
#
# max_delta = max(deltas)
# index = deltas.index(max_delta)
# cdm.all_states[index]
#
# modify2_q_m[item, :] = cdm.all_states[index]
