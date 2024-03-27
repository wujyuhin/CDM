# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
from delta import modify_qvector

logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA

# ============================ 数据准备  ====================================================
# Q矩阵、待修改的Q矩阵、题目数、属性数、作答R矩阵、学生数
q_m = np.loadtxt("../..//data/simulation_q.csv", dtype=int, delimiter=',')
modify_q_m = q_m.copy()
prob_num, know_num = q_m.shape[0], q_m.shape[1]
R = np.array(pd.read_csv("../..//data/simulation_data.csv", index_col=0))
stu_num = R.shape[0]
# ============================ 训练cdm模型 ==================================
cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
cdm.train(epoch=2, epsilon=1e-3)
# ============================ 生成待修改的Q矩阵 ==================================
modify_qvector(modify_q_m, item=0, epsilon=0.05)







