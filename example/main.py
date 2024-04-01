# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
from Qmodify.model.delta import Delta
from Qmodify.model.gamma import Gamma

logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA
'''
# ============================ 数据准备  ====================================================
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

# ============================  加减法数据  ====================================================
# 导入文件夹中FrcSub中的data.txt和q.txt作为作答矩阵与Q矩阵,转成csv格式
# np.savetxt("../data/math2015/FrcSub/q_m.csv", np.loadtxt("../data/math2015/FrcSub/q.txt", dtype=int), delimiter=',', fmt='%d')
# np.savetxt("../data/math2015/FrcSub/data.csv", np.loadtxt("../data/math2015/FrcSub/data.txt", dtype=int), delimiter=',', fmt='%d')
# 数据准备
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# delta_inherit
delta_model = Delta(q_m, R, stu_num, prob_num, know_num,mode='inherit',epsilon=0.05)
modify_q_m1 = delta_model.modify_Q()

delta_model2 = Delta(q_m, R, stu_num, prob_num, know_num,mode='dependence',epsilon=0.05)
modify_q_m2 = delta_model2.modify_Q()

# 将modify1作为Q矩阵使用DINA模型进行参数估计
model = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
model1 = DINA(R, modify_q_m1, stu_num, prob_num, know_num, skip_value=-1)
model1.train(epoch=2, epsilon=0.05)
model2 = DINA(R, modify_q_m2, stu_num, prob_num, know_num, skip_value=-1)
model2.train(epoch=2, epsilon=0.05)
sum(model1.guess)
sum(model2.guess)

sum(model1.slip)
sum(model2.slip)