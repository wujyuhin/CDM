# ============================ 导入必要的包  ====================================================
import numpy as np
from codes.EduCDM import EMDINA as DINA
# ============================  加减法数据进行认知诊断  ==========================================
# 数据准备
# Q矩阵
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')
# 题目数、属性数
prob_num, know_num = q_m.shape[0], q_m.shape[1]
#作答R矩阵
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))
stu_num = R.shape[0]  # 学生数
# cdm估计参数
cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
cdm.train(epoch=2, epsilon=1e-3)
# 题目的guess、slip参数
guess = cdm.guess
slip = cdm.slip
