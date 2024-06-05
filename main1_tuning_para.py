# 研究一:假设检验方法中置信度选择及其可行性和准确性（调参代码）
# 即与其他算法对比，在对比中调整参数，观察结果，最后确定参数范围，然后进行不同参数下的只有SHT算法本身的实验
import time
from tqdm import tqdm
import numpy as np
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R
from code_functions.data_generate.generate import attribute_pattern, state_sample, state_answer
from code_functions.model.hypothesis_skill import Hypothetical_skill
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma
from code_functions.model.metric import PMR, AMR, TPR, FPR
items = 40  # 题目数量
skills = 5  # 知识点数量
students = 1000  # 学生数量
# prob = [0.1, 0.3, 0.4, 0.3, 0.1]  # 题目知识点分布
prob = "frequency"
Qwrong_rate = [0.1, 0.1]
Rwrong_rate = [0.2, 0.2]
sample_mode = "frequency"
# sample_mode = "normal"

amr, pmr, tpr, fpr = [], [], [], []
t = []
np.random.seed(0)
for i in tqdm(range(1)):
    Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
    wrong_Q = generate_wrong_Q(Q, Rwrong_rate)['Q_wrong']  # 生成错误率的Q矩阵
    states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    # states_samples = state_sample(states, num=students, method="normal", mu_skills=2, sigma_skills=0.5)  # 从掌握模式中抽样
    states_samples = state_sample(states, num=students, method=sample_mode,mu_skills=2,sigma_skills=1)  # 从掌握模式中抽样
    R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    R = generate_wrong_R(R, wrong_rate=Qwrong_rate)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大

    ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
    delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.001,mode='dependence')
    gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
    # modify_q_m1 = ht.modify_Q(mode='loop')
    t1 = time.time()
    modify_q_m1 = ht.modify_Q_method3(mode='loop')
    t2 = time.time()
    modify_q_m2 = delta.modify_Q()
    t3 = time.time()
    modify_q_m3 = gamma.modify_Q()
    t4 = time.time()
    t.append([t2-t1,t3-t2,t4-t3])
    pmr.append([PMR(Q, modify_q_m1), PMR(Q, modify_q_m2), PMR(Q, modify_q_m3),PMR(Q, wrong_Q)])
    amr.append([AMR(Q, modify_q_m1), AMR(Q, modify_q_m2), AMR(Q, modify_q_m3),AMR(Q, wrong_Q)])
    tpr.append([TPR(Q, wrong_Q, modify_q_m1), TPR(Q, wrong_Q, modify_q_m2), TPR(Q, wrong_Q, modify_q_m3)])
    fpr.append([FPR(Q, wrong_Q, modify_q_m1), FPR(Q, wrong_Q, modify_q_m2), FPR(Q, wrong_Q, modify_q_m3)])

print("参数为：")
print("学生数量：", students)
print("知识点数量：", skills)
print("题目数量：", items)
print("知识点分布：", prob)
print("学生掌握模式抽样模式：", sample_mode)
print("Q矩阵错误率：", Qwrong_rate)
print("R矩阵错误率：", Rwrong_rate)
print("平均修改时间：",np.mean(t,axis=0))
print('修改Q矩阵估计的PMR参数：', np.mean(pmr, axis=0))
print('修改Q矩阵估计的AMR参数：', np.mean(amr, axis=0))
print('修改Q矩阵估计的TPR参数：', np.mean(tpr, axis=0))
print('修改Q矩阵估计的FPR参数：', np.mean(fpr, axis=0))