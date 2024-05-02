# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
from code_functions.model.hypothetical import Hypothetical
from code_functions.model.metric import PMR, AMR, TPR, FPR
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, state_sample, attribute_pattern, state_answer

'''
# ============================ 模拟数据准备  ====================================================
q_m = np.loadtxt("../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(pd.read_csv("../data/math2015/simulation/simulation_data.csv", index_col=0))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# ============================ delta法 ==================================
delta_model = Delta(q_m, R, stu_num, prob_num, know_num,epsilon=0.05)
modify_q_m1 = delta_model.modify_Q()
modify_q_m2 = delta_model.modify_Q_inherit()

# ============================ gamma法 ==================================
# gamma_model = Gamma(q_m, R, stu_num, prob_num, know_num,threshold_g=0.2,threshold_s=0.2,threshold_es=0.2)
# modify_q_m2 = gamma_model.modify_Q()
'''

np.random.seed(0)
skills = 4
items = 24
students = 2000
amr, pmr, tpr, fpr = [], [], [], []
wrong = 0.1

# Q = generate_Q(items, skills, probs=[0.1, 0.2, 0.3,0.4])
Q = generate_Q(items, skills, probs=[0.2, 0.6, 0.2,0])
result = generate_wrong_Q(Q, wrong)
wrong_Q = result['Q_wrong']
states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
# states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
states_samples = state_sample(states, num=students, method="assign",set_skills=2)  # 从掌握模式中抽样
answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
# hypothesis
hypothesis_model = Hypothetical(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills, mode='loop')
modify_q_m1 = hypothesis_model.modify_Q(alpha=0.01)  # 修正Q矩阵
# delta
delta_model = Delta(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills, mode='inherit', epsilon=0.05)
modify_q_m2 = delta_model.modify_Q()  # Q矩阵
# gamma
gamma_model = Gamma(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
modify_q_m3 = gamma_model.modify_Q()  # Q矩阵

# 计算AMR,PMR,TPR,FPR
pmr.append([PMR(Q, modify_q_m1), PMR(Q, modify_q_m2), PMR(Q, modify_q_m3)])
amr.append([AMR(Q, modify_q_m1), AMR(Q, modify_q_m2), AMR(Q, modify_q_m3)])
tpr.append([TPR(Q, wrong_Q, modify_q_m1), TPR(Q, wrong_Q, modify_q_m2), TPR(Q, wrong_Q, modify_q_m3)])
fpr.append([FPR(Q, wrong_Q, modify_q_m1), FPR(Q, wrong_Q, modify_q_m2), FPR(Q, wrong_Q, modify_q_m3)])

print('修改Q矩阵估计的PMR参数：', np.mean(pmr, axis=0))
print('修改Q矩阵估计的AMR参数：', np.mean(amr, axis=0))
print('修改Q矩阵估计的TPR参数：', np.mean(tpr, axis=0))
print('修改Q矩阵估计的FPR参数：', np.mean(fpr, axis=0))