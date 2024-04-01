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


# ============================  加减法数据  ====================================================
# 数据准备
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# 边发现边修正
delta_model = Delta(q_m, R, stu_num, prob_num, know_num,mode='inherit',epsilon=0.05)
modify_q_m1 = delta_model.modify_Q() # Q矩阵

# 发现完所有的再修正
delta_model2 = Delta(q_m, R, stu_num, prob_num, know_num,mode='dependence',epsilon=0.05)
modify_q_m2 = delta_model2.modify_Q() # Q矩阵

# 对上述两种Q矩阵使用DINA模型进行参数估计
model = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
model1 = DINA(R, modify_q_m1, stu_num, prob_num, know_num, skip_value=-1)

model1.train(epoch=2, epsilon=0.05)
model2 = DINA(R, modify_q_m2, stu_num, prob_num, know_num, skip_value=-1)
model2.train(epoch=2, epsilon=0.05)

print('边发现边修改Q矩阵估计的平均guess参数：',sum(model1.guess))
print('发现完所有的再修改Q矩阵估计的平均guess参数：',sum(model2.guess))


print('边发现边修改Q矩阵估计的平均slip参数：',sum(model1.slip))
print('发现完所有的再修改Q矩阵估计的平均slip参数：',sum(model2.slip))