# 研究二:SHT算法与其他算法的比较函数，参数设定
# sht算法的参数设定：alpha=0.01,
# delta法参数设定：epsilon=0.01,
# gamma法参数设定：threshold_g=0.2,threshold_s=0.2,threshold_es=0.2

import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R
from code_functions.data_generate.generate import attribute_pattern, state_sample, state_answer
from code_functions.model.hypothesis_skill import Hypothetical_skill
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
from code_functions.model.metric import PMR, AMR, TPR, FPR

t_start = time.time()
items = 40  # 题目数量
skills = 5  # 知识点数量
q_wrong_rate = 0.2
prob = "frequency"
sample_mode = "frequency"
np.random.seed(0)
result_dict = {}
time_ht, time_delta, time_gamma = [], [], []
for Rwrong_rate in [0.1, 0.15, 0.2]:
    students_dict = {}
    for students in [100, 300, 1000, 2000]:
        Qwrong_rate = [q_wrong_rate, q_wrong_rate]
        # ======== 初始化  ===============
        wrong_pmr, wrong_amr = [], []  # 错误Q矩阵的pmr,amr
        amr, pmr, tpr, fpr = [], [], [], []  # 修改Q矩阵的pmr,amr,tpr,fpr
        amr_delta, pmr_delta, tpr_delta, fpr_delta = [], [], [], []  # 修改Q矩阵的pmr,amr,tpr,fpr
        amr_gamma, pmr_gamma, tpr_gamma, fpr_gamma = [], [], [], []
        for i in tqdm(range(100), desc=f"Rwrong_rate_{Rwrong_rate}_students_{students}"):
            # 对于假设检验方法，置信度选择应该对统一标准的Q矩阵和R矩阵
            Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
            wrong_Q = generate_wrong_Q(Q, Qwrong_rate)['Q_wrong']  # 生成错误率的Q矩阵
            states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
            states_samples = state_sample(states, num=students, method=sample_mode)  # 从掌握模式中抽样
            R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
            R = generate_wrong_R(R, wrong_rate=Rwrong_rate)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
            # 开始修正Q矩阵
            t1 = time.time()
            ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
            modify_q = ht.modify_Q_method3(mode='loop')
            t2 = time.time()
            delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.01, mode='dependence')
            modify_q_delta = delta.modify_Q()
            t3 = time.time()
            gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
            modify_q_gamma = gamma.modify_Q()
            t4 = time.time()
            # 计算指标 1.错误Q矩阵的PMR,AMR 2.修改Q矩阵的PMR,AMR,TPR,FPR
            wrong_pmr.append(PMR(Q, wrong_Q))
            wrong_amr.append(AMR(Q, wrong_Q))
            pmr.append(PMR(Q, modify_q))
            amr.append(AMR(Q, modify_q))
            tpr.append(TPR(Q, wrong_Q, modify_q))
            fpr.append(FPR(Q, wrong_Q, modify_q))
            pmr_delta.append(PMR(Q, modify_q_delta))
            amr_delta.append(AMR(Q, modify_q_delta))
            tpr_delta.append(TPR(Q, wrong_Q, modify_q_delta))
            fpr_delta.append(FPR(Q, wrong_Q, modify_q_delta))
            pmr_gamma.append(PMR(Q, modify_q_gamma))
            amr_gamma.append(AMR(Q, modify_q_gamma))
            tpr_gamma.append(TPR(Q, wrong_Q, modify_q_gamma))
            fpr_gamma.append(FPR(Q, wrong_Q, modify_q_gamma))
            # time
            time_ht.append(t2 - t1)
            time_delta.append(t3 - t2)
            time_gamma.append(t4 - t3)
        # print(f"已完成Qwrong{q_wrong_rate},Rwrong{Rwrong_rate},students{students}的实验")
        result = np.array(
            [wrong_pmr, pmr, pmr_delta, pmr_gamma, wrong_amr, amr, amr_delta, amr_gamma, tpr, tpr_delta, tpr_gamma, fpr,
             fpr_delta, fpr_gamma])
        students_dict["students_" + str(students)] = result
    result_dict["Rwrong_rate_" + str(Rwrong_rate)] = students_dict

t_end = time.time()
print("time cost:",t_end-t_start)
# 保存result_dict字典为pkl
import pickle
with open(f'report/result/research2/result_Qwrong_rate_{q_wrong_rate}.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
# 计算平均值
a = np.array([result_dict['Rwrong_rate_0.1']['students_100'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.1']['students_300'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.1']['students_1000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.1']['students_2000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.1']['students_100'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.1']['students_300'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.1']['students_1000'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.1']['students_2000'].mean(axis=1)[4:8],
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_100'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_300'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_1000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_2000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_100'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_300'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_1000'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.1']['students_2000'].mean(axis=1)[11:])
              ]).transpose()
b = np.array([result_dict['Rwrong_rate_0.15']['students_100'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.15']['students_300'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.15']['students_1000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.15']['students_2000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.15']['students_100'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.15']['students_300'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.15']['students_1000'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.15']['students_2000'].mean(axis=1)[4:8],
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_100'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_300'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_1000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_2000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_100'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_300'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_1000'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.15']['students_2000'].mean(axis=1)[11:])
              ]).transpose()

c = np.array([result_dict['Rwrong_rate_0.2']['students_100'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.2']['students_300'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.2']['students_1000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.2']['students_2000'].mean(axis=1)[0:4],
              result_dict['Rwrong_rate_0.2']['students_100'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.2']['students_300'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.2']['students_1000'].mean(axis=1)[4:8],
              result_dict['Rwrong_rate_0.2']['students_2000'].mean(axis=1)[4:8],
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_100'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_300'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_1000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_2000'].mean(axis=1)[8:11]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_100'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_300'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_1000'].mean(axis=1)[11:]),
              np.append(np.nan, result_dict['Rwrong_rate_0.2']['students_2000'].mean(axis=1)[11:])
              ]).transpose()


# result_array = np.concatenate((a, b, c), axis=0)
# pd.DataFrame(result_array).to_csv(f'report/result/research2/result_Qwrong_rate_{q_wrong_rate}.csv')
# print("已完q_wrong_rate为", q_wrong_rate, "的实验")
