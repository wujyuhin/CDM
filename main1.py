# 研究一:假设检验方法中置信度选择及其可行性和准确性
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from codes.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R
from codes.data_generate.generate import attribute_pattern, state_sample, state_answer
from codes.model.SelectHypothesisTest import SelectHypothesisTest as SHT
from codes.model.delta import Delta
from codes.model.gamma import Gamma
from codes.model.metric import PMR, AMR, TPR, FPR
items = 40  # 题目数量
skills = 5  # 知识点数量
students = 1000  # 学生数量
# prob = [0.1, 0.3, 0.4, 0.3, 0.1]  # 题目知识点分布
prob = "frequency"
Rwrong_rate = 0.15
sample_mode = "frequency"

# np.random.seed(0)
result_dict = {}
t1 = time.time()

for q_wrong_rate in [0.05, 0.1, 0.15,0.2]:
    Qwrong_rate = [q_wrong_rate, q_wrong_rate]
    alpha_dict = {"alpha_0.01": [],"alpha_0.05": [],"alpha_0.1": []}
    wrong_pmr, wrong_amr = [], []  # 错误Q矩阵的pmr,amr
    amr, pmr, tpr, fpr = [], [], [], []  # 修改Q矩阵的pmr,amr,tpr,fpr
    pmr_up, amr_up = [], []  # pmr,amr的提升比例
    for i in tqdm(range(1), desc=f"Qwrong_rate_{q_wrong_rate},alpha为0.01,0.05,0.1"):
        np.random.seed(i)
        # 对于假设检验方法，置信度选择应该对统一标准的Q矩阵和R矩阵
        Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
        wrong_Q = generate_wrong_Q(Q, Qwrong_rate)['Q_wrong']  # 生成错误率的Q矩阵
        states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
        states_samples = state_sample(states, num=students, method=sample_mode)  # 从掌握模式中抽样
        R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
        R = generate_wrong_R(R, wrong_rate=Rwrong_rate)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
        for alpha in [0.01,0.05,0.1]:
            ht = SHT(wrong_Q, R, students, items, skills, alpha)
            modify_q = ht.modify_Q(mode='loop')
            # 计算指标 1.错误Q矩阵的PMR,AMR 2.修改Q矩阵的PMR,AMR,TPR,FPR
            alpha_dict["alpha_"+str(alpha)].append([PMR(Q, wrong_Q),PMR(Q, modify_q),PMR(Q, modify_q)-PMR(Q, wrong_Q),
                                                    AMR(Q, wrong_Q),AMR(Q, modify_q),AMR(Q, modify_q)-AMR(Q, wrong_Q),
                                                    TPR(Q, wrong_Q, modify_q),FPR(Q, wrong_Q, modify_q)])
            # 将wrong_pmr,pmr,pmr_up,wrong_amr,amr,amr_up,tpr,fpr转化为numpy数组,并合并成同一个二维数组
            # result = np.array([wrong_pmr,pmr,pmr_up,wrong_amr,amr,amr_up,tpr,fpr])
            # alpha_dict["alpha_"+str(alpha)] = result
    result_dict["Qwrong_rate_"+str(q_wrong_rate)] = alpha_dict
# 保存result_dict字典为pkl

t2 = time.time()
print("time cost:",t2-t1)


# 计算平均值
result_array = np.array([np.array(result_dict['Qwrong_rate_0.05']['alpha_0.01']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.05']['alpha_0.05']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.05']['alpha_0.1']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.1']['alpha_0.01']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.1']['alpha_0.05']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.1']['alpha_0.1']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.15']['alpha_0.01']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.15']['alpha_0.05']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.15']['alpha_0.1']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.2']['alpha_0.01']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.2']['alpha_0.05']).mean(axis=0),
                         np.array(result_dict['Qwrong_rate_0.2']['alpha_0.1']).mean(axis=0)])

# import pickle
# with open(f'report/result/research1/method3/result_dict_Rwrong_rate_{Rwrong_rate}.pkl', 'wb') as f:
#     pickle.dump(result_dict, f)
# pd.DataFrame(result_array).to_csv(f'report/result/research1/method3/result_Rwrong_rate_{Rwrong_rate}.csv')
# print("Rwrong_rate:",Rwrong_rate,"修正Q矩阵(查看是否是随机数问题)")