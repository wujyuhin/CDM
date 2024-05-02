# ============================ 导入必要的包  ====================================================
import numpy as np
import logging
from code_functions.model.hypothetical import Hypothetical
from code_functions.EduCDM import EMDINA as DINA
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, generate_wrong_R
from code_functions.data_generate.generate import attribute_pattern, state_sample, state_answer
from code_functions.model.metric import PMR, AMR, TPR,FPR
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)

# ============================  加减法数据  ====================================================
# 数据准备
# q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
# prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
# R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
# stu_num = R.shape[0]  # 学生数


# 生成数据
np.random.seed(0)
skills = 3
items = 24
students = 1000
quality = [0.2,0.2]
amr, pmr, tpr, fpr = [], [], [], []
for i in tqdm(range(10)):
    wrong = 0.1
    Q = generate_Q(items, skills, probs=[0.3, 0.3, 0.4])
    result = generate_wrong_Q(Q, wrong)
    wrong_Q = result['Q_wrong']
    states = np.concatenate((np.zeros((1,skills)),attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    answer = generate_wrong_R(answer, wrong_rate=quality)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    hypothesis_model = Hypothetical(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills)  # 实例化
    # hypothesis_model2 = Hypothetical(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills,mode='loop')  # 实例化
    modify_q_m1 = hypothesis_model.modify_Q(mode='loop',alpha=0.01)  # 循环修正Q矩阵
    modify_q_m2 = hypothesis_model.modify_Q(mode='no_loop',alpha=0.01)  # 非循环修正Q矩阵


    pmr.append([PMR(Q, modify_q_m1), PMR(Q, modify_q_m2)])
    amr.append([AMR(Q, modify_q_m1), AMR(Q, modify_q_m2)])
    tpr.append([TPR(Q, wrong_Q, modify_q_m1), TPR(Q, wrong_Q, modify_q_m2)])
    fpr.append([FPR(Q, wrong_Q, modify_q_m1), FPR(Q, wrong_Q, modify_q_m2)])

print('修改Q矩阵估计的PMR参数：',np.mean(pmr,axis=0))
print('修改Q矩阵估计的AMR参数：',np.mean(amr,axis=0))
print('修改Q矩阵估计的TPR参数：',np.mean(tpr,axis=0))
print('修改Q矩阵估计的FPR参数：',np.mean(fpr,axis=0))


