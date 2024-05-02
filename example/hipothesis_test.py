# ============================ 导入必要的包  ====================================================
import numpy as np
import logging
from code_functions.model.hypothetical import Hypothetical
from code_functions.EduCDM import EMDINA as DINA
from code_functions.data_generate.generate import generate_Q, generate_wrong_Q, state_sample, attribute_pattern, state_answer
from code_functions.model.metric import PMR, AMR, TPR,FPR
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)

# ============================  加减法数据  ====================================================
# 数据准备
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# # ============================  边发现边修正  ====================================================
# # 模型实例化
# hypothesis_model = Hypothetical(q_m, R, stu_num, prob_num, know_num,mode='loop')
# # 对输入的实例进行修正
# modify_q_m1 = hypothesis_model.modify_Q(alpha=0.1)  # Q矩阵
#
# # 对上述两种Q矩阵使用DINA模型进行参数估计
# model1 = DINA(R, modify_q_m1, stu_num, prob_num, know_num, skip_value=-1)
# model1.train(epoch=2, epsilon=0.05)
# model2 = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
# model2.train(epoch=2, epsilon=0.05)
#
# print('修改Q矩阵估计的平均guess参数：',sum(model1.guess)/len(model1.guess))
# print('原本Q矩阵估计的平均guess参数：',sum(model2.guess)/len(model2.guess))
#
# print('修改Q矩阵估计的平均slip参数：',sum(model1.slip)/len(model1.slip))
# print('原本Q矩阵估计的平均slip参数：',sum(model2.slip)/len(model2.slip))

# 生成数据
np.random.seed(0)
skills = 3
items = 24
students = 1000
amr, pmr, tpr, fpr = [], [], [], []
for i in tqdm(range(200)):
    wrong = 0.05
    Q = generate_Q(items, skills, probs=[0.3, 0.3, 0.4])
    result = generate_wrong_Q(Q, wrong)
    wrong_Q = result['Q_wrong']
    states = np.concatenate((np.zeros((1,skills)),attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    # 模型实例化
    hypothesis_model = Hypothetical(q_m=wrong_Q, R=answer, stu_num=students, prob_num=items, know_num=skills,mode='loop')
    modify_q_m1 = hypothesis_model.modify_Q(alpha=0.01)  # 修正Q矩阵

    # 计算AMR,PMR,TPR,FPR
    pmr.append(PMR(Q, modify_q_m1))
    amr.append(AMR(Q, modify_q_m1))
    tpr.append(TPR(Q,wrong_Q, modify_q_m1))
    fpr.append(FPR(Q,wrong_Q, modify_q_m1))
print('修改Q矩阵估计的PMR参数：',np.mean(pmr))
print('修改Q矩阵估计的AMR参数：',np.mean(amr))
print('修改Q矩阵估计的TPR参数：',np.mean(tpr))
print('修改Q矩阵估计的FPR参数：',np.mean(fpr))
