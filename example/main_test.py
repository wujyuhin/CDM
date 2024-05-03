import numpy as np
from tqdm import tqdm
import time
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R, state_sample, attribute_pattern, state_answer
from code_functions.model.hypothetical import Hypothetical
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
from code_functions.model.metric import PMR, AMR, TPR, FPR
from code_functions.EduCDM import EMDINA as DINA

np.random.seed(0)
# students = [300, 500, 1000]  # 生成学生数量
# skills_items_probs = [[3, 24, [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]],
#                       [4, 32, [[0.2, 0.6, 0.2, 0], [0, 0.2, 0.6, 0.2]]]]
# [5,40,[[0.1, 0.4, 0.3, 0.2, 0],[0, 0.2, 0.3, 0.4, 0.1]]]
# ]  # 生成知识点数、题目数、知识点数量分布
# Q_wrong_rate = [0.05, 0.1, 0.15]  # 生成Q矩阵错误率
# qualities = [[0.1 * i, 0.1 * j] for i in range(4) for j in range(4)]  # 生成题目质量
# qualities = [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
# sample_modes = ["uniform_mode", "normal", "assign"]
# sample_modes_para = {"uniform_mode": None, "normal": [2, 1], "assign": 1}  # normal:均值为2，方差为1 assign:指定抽样只掌握2个知识点
# 参数设置
students = [1000]  # 生成学生数量
skills_items_probs = [[4, 32, [[0.4, 0.5, 0.1, 0]]]]
# skills_items_probs = [[3, 24, [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]]]
Q_wrong_rate = [0.1]  # 生成Q矩阵错误率
qualities = [[0.1, 0.1]]  # 生成题目质量
sample_modes = ["uniform_mode"]  # 生成抽样模式
sample_modes_para = {"uniform_mode": None, "normal": [2, 0.5], "assign": 1}  # normal:均值为2，方差为1 assign:指定抽样只掌握2个知识点

t1 = time.time()
dataset = []
data = {}
amr, pmr, tpr, fpr = [], [], [], []
time_cost = []
np.random.seed(0)
for i in tqdm(range(1)):  # 每类数据生成1次
    for student in students:
        # t1 = time.time()
        for skills, items, probs in skills_items_probs:
            for prob in probs:
                for wrong in Q_wrong_rate:
                    tt1 = time.time()
                    for quality in qualities:
                        # t1 = time.time()
                        for mode in sample_modes:
                            Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
                            wrong_Q_object = generate_wrong_Q(Q, wrong)  # 生成错误率的Q矩阵
                            wrong_Q = wrong_Q_object['Q_wrong']
                            # wrong_Q = np.load('wrong_Q_4skill.npy')
                            # 保存wrong_Q numpy文件
                            # np.save('wrong_Q_4skill.npy', wrong_Q)
                            states = np.concatenate(
                                (np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
                            if mode == "normal":
                                states_samples = state_sample(states, num=student, method=mode,
                                                              mu_skills=sample_modes_para[mode][0],
                                                              sigma_skills=sample_modes_para[mode][1])
                            elif mode == "assign":
                                states_samples = state_sample(states, num=student, method=mode,
                                                              set_skills=sample_modes_para[mode])
                            else:
                                states_samples = state_sample(states, num=student, method=mode)  # 从掌握模式中抽样
                            answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples,Q=Q)  # 根据掌握模式生成作答情况
                            answer = generate_wrong_R(answer, wrong_rate=quality)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大

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

                            # ================================= metric =================================
                            pmr.append([PMR(Q, modify_q_hypothesis), PMR(Q, modify_q_delta), PMR(Q, modify_q_gamma)])
                            amr.append([AMR(Q, modify_q_hypothesis), AMR(Q, modify_q_delta), AMR(Q, modify_q_gamma)])
                            tpr.append([TPR(Q, wrong_Q, modify_q_hypothesis), TPR(Q, wrong_Q, modify_q_delta), TPR(Q, wrong_Q, modify_q_gamma)])
                            fpr.append([FPR(Q, wrong_Q, modify_q_hypothesis), FPR(Q, wrong_Q, modify_q_delta), FPR(Q, wrong_Q, modify_q_gamma)])
                            time_cost.append([t2 - t1, t3 - t2, t4 - t3])

print('参数为：')
print('学生数量：', students)
print('知识点数、题目数、知识点数量分布：', skills_items_probs)
print('Q矩阵错误率：', Q_wrong_rate)
print('题目质量：', qualities)
print('抽样模式：', sample_modes)
print('normal抽样均值和方差：', sample_modes_para["normal"])
print('修改Q矩阵估计的PMR参数：', np.mean(pmr, axis=0))
print('修改Q矩阵估计的AMR参数：', np.mean(amr, axis=0))
print('修改Q矩阵估计的TPR参数：', np.mean(tpr, axis=0))
print('修改Q矩阵估计的FPR参数：', np.mean(fpr, axis=0))




print('即错误Q矩阵具有冗余的1，将其修正为0')
for i,j in wrong_Q_object['wrong_set_01']:
    flag = 0
    i,j = int(i),int(j)
    if modify_q_hypothesis[i,j] == Q[i,j]:
        flag += 1
        print('修正了',i,j)
    else:
        print('未修正',i,j)

print('即错误Q矩阵属性缺失的1，将其补为1')
for i,j in wrong_Q_object['wrong_set_10']:
    flag = 0
    i,j = int(i),int(j)
    if modify_q_hypothesis[i,j] == Q[i,j]:
        flag += 1
        print('修正了',i,j)
    else:
        print('未修正',i,j)

