""" 研究二实例：本文件用于对比delaTorre使用的Q矩阵与我们使用的Q矩阵的差异 """
import random
import numpy as np
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R
from code_functions.data_generate.generate import attribute_pattern, state_sample, state_answer_gs
from code_functions.model.hypothesis_skill import Hypothetical_skill
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
import pandas as pd


def delaTorre(skills, items, students, R, wrong_Q):
    ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
    delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.05)
    gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
    modify_q_m1 = ht.modify_Q(mode='loop', alpha=0.01)
    modify_q_m2 = delta.modify_Q()
    modify_q_m3 = gamma.modify_Q()
    return modify_q_m1, modify_q_m2, modify_q_m3


Q = pd.read_excel('./data/delaTorre/q_m.xlsx', header=None).values  # 真实Q矩阵
students = 1000  # 学生数量
items = Q.shape[0]  # 题目数量
skills = Q.shape[1]  # 知识点数量
gs = [0.2, 0.2]

states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
states_samples = state_sample(states, num=students, method="uniform_mode", mu_skills=3, sigma_skills=1)  # 从掌握模式中抽样
np.random.seed(0)
R = np.apply_along_axis(state_answer_gs, axis=1, arr=states_samples, Q=Q, g=gs[0], s=gs[1])  # 根据掌握模式生成作答情况
wrong_Q = Q.copy()

print("======================= item1 ========================")
wrong_Q[0, 0] = 0
wrong_Q[0, 1] = 1
item1 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[0, :])
print('wrong_Q', wrong_Q[0, :])
print('ht修改', item1[0][0, :])
print('delta修改', item1[1][0, :])
print('gamma修改', item1[2][0, :])

print("======================= item2 ========================")
wrong_Q = Q.copy()
wrong_Q[0, 1] = 1
item2 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[0, :])
print('wrong_Q', wrong_Q[0, :])
print('ht修改', item2[0][0, :])
print('delta修改', item2[1][0, :])
print('gamma修改', item2[2][0, :])

# item3
print("======================= item3 ========================")
wrong_Q = Q.copy()
wrong_Q[10, 0] = 0
wrong_Q[10, 2] = 1
item3 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[10, :])
print('wrong_Q', wrong_Q[10, :])
print('ht修改', item3[0][10, :])
print('delta修改', item3[1][10, :])
print('gamma修改', item3[2][10, :])

# item4
print("======================= item4 ========================")
wrong_Q = Q.copy()
wrong_Q[10, 0] = 0
item4 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[10, :])
print('wrong_Q', wrong_Q[10, :])
print('ht修改', item4[0][10, :])
print('delta修改', item4[1][10, :])
print('gamma修改', item4[2][10, :])

# item5
print("======================= item5 ========================")
wrong_Q = Q.copy()
wrong_Q[10, 2] = 1
item5 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[10, :])
print('wrong_Q', wrong_Q[10, :])
print('ht修改', item5[0][10, :])
print('delta修改', item5[1][10, :])
print('gamma修改', item5[2][10, :])

# item6
print("======================= item6 ========================")
wrong_Q = Q.copy()
wrong_Q[20, 0] = 0
item6 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item6[0][20, :])
print('delta修改', item6[1][20, :])
print('gamma修改', item6[2][20, :])

# item7
print("======================= item7 ========================")
wrong_Q = Q.copy()
wrong_Q[20, 0] = 0
wrong_Q[20, 1] = 0
item7 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item7[0][20, :])
print('delta修改', item7[1][20, :])
print('gamma修改', item7[2][20, :])

# item8
print("======================= item8 ========================")
wrong_Q = Q.copy()
wrong_Q[20, 0] = 0
wrong_Q[20, 3] = 1
item8 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item8[0][20, :])
print('delta修改', item8[1][20, :])
print('gamma修改', item8[2][20, :])

# item9
print("======================= item9 ========================")
wrong_Q = Q.copy()
wrong_Q[20, 0] = 0
wrong_Q[20, 1] = 0
wrong_Q[20, 3] = 1
item9 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item9[0][20, :])
print('delta修改', item9[1][20, :])
print('gamma修改', item9[2][20, :])

# item10
print("======================= item10 ========================")
wrong_Q = Q.copy()
wrong_Q[20, 0] = 0
wrong_Q[20, 1] = 0
wrong_Q[20, 3] = 1
wrong_Q[20, 4] = 1
item10 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item10[0][20, :])
print('delta修改', item10[1][20, :])
print('gamma修改', item10[2][20, :])

# item11
print("======================= item11 ========================")
wrong_Q = Q.copy()
wrong_Q[0, 1] = 1
wrong_Q[10, 1] = 0
wrong_Q[10, 2] = 0
wrong_Q[20, 0] = 0
wrong_Q[20, 3] = 1
item11 = delaTorre(skills, items, students, R, wrong_Q)
print('原始Q_0', Q[0, :])
print('wrong_Q', wrong_Q[0, :])
print('ht修改', item11[0][0, :])
print('delta修改', item11[1][0, :])
print('gamma修改', item11[2][0, :])

print('原始Q_10', Q[10, :])
print('wrong_Q', wrong_Q[10, :])
print('ht修改', item11[0][10, :])
print('delta修改', item11[1][10, :])
print('gamma修改', item11[2][10, :])

print('原始Q_20', Q[20, :])
print('wrong_Q', wrong_Q[20, :])
print('ht修改', item11[0][20, :])
print('delta修改', item11[1][20, :])
print('gamma修改', item11[2][20, :])
