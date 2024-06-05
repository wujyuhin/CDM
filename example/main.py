# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
import time
import pickle
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
from code_functions.model.hypothetical import Hypothetical
from code_functions.model.metric import PMR, AMR, TPR, FPR
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q,generate_wrong_R
from code_functions.data_generate.generate import state_sample, attribute_pattern, state_answer

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
skills = 4  # 知识点数
skills_probs = [0.2, 0.6, 0.2,0]  # 考察知识点数量分布
items = 24  # 题目数
students = 2000  # 学生数
# amr, pmr, tpr, fpr = [], [], [], []
wrong = 0.1  # Q矩阵错误率
quality = 0.2  # 题目质量

# 多种算法比较性能
# Q = generate_Q(items, skills, probs=skills_probs)
# result = generate_wrong_Q(Q, wrong_rate=wrong)
# wrong_Q = result['Q_wrong']
# states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
# states_samples = state_sample(states, num=students, method="assign",set_skills=2)  # 从掌握模式中抽样
# answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
# answer = generate_wrong_R(answer, wrong_rate=quality)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
with open('../data/dataset_error.pkl', 'rb') as f:
    dataset = pickle.load(f)
students = [300, 500, 1000]  # 生成学生数量
skills_items_probs = [[3, 24, [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]],
                      [4, 32, [[0.2, 0.6, 0.2, 0], [0, 0.2, 0.6, 0.2]]]]# 生成知识点数、题目数、知识点数量分布
Q_wrong_rate = [0.05, 0.1, 0.15]  # 生成Q矩阵错误率
qualities = [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]  # 生成题目质量
sample_modes = ["uniform_mode", "normal", "assign"]  # 生成抽样模式
sample_modes_para = {"uniform_mode": None, "normal": [2, 1], "assign": 1}  # normal:均值为2，方差为1 assign:指定抽样只掌握2个知识点
metric_set = {}
index = 0
for data in tqdm.tqdm(dataset):
    metric_student_num = {}
    for student in students:
        metric_skill = {}
        for skills, items, probs in skills_items_probs:
            metric_prob = {}
            for prob in probs:
                metric_wrong = {}
                for wrong in Q_wrong_rate:
                    metric_quality = {}
                    for quality in qualities:
                        metric_mode = {}
                        for mode in sample_modes:
                            t1 = time.time()
                            a = data[f"{student}_{skills}_{items}_{prob}_{wrong}_{quality}_{mode}"]
                            Q = a['Q']
                            wrong_Q = a['Q_wrong']
                            answer = a['answer']
                            # ================================= hypothesis =================================
                            hypothesis_model = Hypothetical(q_m=wrong_Q, R=answer, stu_num=student, prob_num=items, know_num=skills)  # 实例化
                            modify_q_hypothesis = hypothesis_model.modify_Q(mode='loop', alpha=0.01)  # 循环修正Q矩阵
                            t2 = time.time()
                            # ================================= delta =================================
                            delta_model = Delta(q_m=wrong_Q, R=answer, stu_num=student, prob_num=items, know_num=skills, mode='inherit', epsilon=0.05)
                            modify_q_delta = delta_model.modify_Q()
                            t3 = time.time()
                            # ================================= gamma =================================
                            gamma_model = Gamma(q_m=wrong_Q, R=answer, stu_num=student, prob_num=items, know_num=skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
                            modify_q_gamma = gamma_model.modify_Q()
                            t4 = time.time()
                            # 计算AMR,PMR,TPR,FPR
                            pmr = [PMR(Q, modify_q_hypothesis), PMR(Q, modify_q_delta), PMR(Q, modify_q_gamma)]
                            amr = [AMR(Q, modify_q_hypothesis), AMR(Q, modify_q_delta), AMR(Q, modify_q_gamma)]
                            tpr = [TPR(Q, wrong_Q, modify_q_hypothesis), TPR(Q, wrong_Q, modify_q_delta), TPR(Q, wrong_Q, modify_q_gamma)]
                            fpr = [FPR(Q, wrong_Q, modify_q_hypothesis), FPR(Q, wrong_Q, modify_q_delta), FPR(Q, wrong_Q, modify_q_gamma)]
                            # metric = {}
                            # metric['pmr'] = pmr
                            # metric['amr'] = amr
                            # metric['tpr'] = tpr
                            # metric['fpr'] = fpr
                            # metric['time'] = [t2-t1, t3-t2, t4-t3]
                            # metric_set[f"data{index}_{student}_{skills}_{items}_{prob}_{wrong}_{quality}_{mode}"] = metric
                            metric_mode[f"{mode}"] = {'pmr':pmr, 'amr':amr, 'tpr':tpr, 'fpr':fpr, 'time':[t2-t1, t3-t2, t4-t3]}
                        metric_quality[f"{quality}"] = metric_mode
                    metric_wrong[f"{wrong}"] = metric_quality
                metric_prob[f"{prob}"] = metric_wrong
            metric_skill[f"{skills}_{items}"] = metric_prob
        metric_student_num[f"{student}"] = metric_skill
    metric_set[f"data{index}"] = metric_student_num
    index += 1







# 保存dataset
with open('all_metrics.pkl', "wb") as f:
    pickle.dump(metric_set, f)



