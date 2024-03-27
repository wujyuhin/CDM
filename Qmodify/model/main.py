# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
from delta import Delta
logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA

# ============================ 数据准备  ====================================================
q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))  #作答R矩阵
stu_num = R.shape[0]  # 学生数
# ============================ 训练cdm模型 ==================================
cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
cdm.train(epoch=2, epsilon=1e-3)
# ============================ delta法 ==================================
delta = Delta(q_m, R, stu_num, prob_num, know_num)
for item in tqdm(range(prob_num)):
    modify_q_m = delta.modify_qvector(modify_q_m, item, epsilon=0)







