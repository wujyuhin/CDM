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

# ============================ 数据准备  ====================================================
q_m = np.loadtxt("../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(pd.read_csv("../data/math2015/simulation/simulation_data.csv", index_col=0))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# ============================ delta法 ==================================
delta_model = Delta(q_m, R, stu_num, prob_num, know_num,epsilon=0.05)
modify_q_m1 = delta_model.modify_Q()

# ============================ gamma法 ==================================
gamma_model = Gamma(q_m, R, stu_num, prob_num, know_num,threshold_g=0.2,threshold_s=0.2,threshold_es=0.2)
modify_q_m2 = gamma_model.modify_Q()







