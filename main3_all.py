# 研究三 实例验证 调用R代码计算拟合度
import time

import numpy as np
import pandas as pd
from code_functions.EduCDM import EMDINA as DINA
from code_functions.model.hypothesis_skill import Hypothetical_skill
from code_functions.model.delta import Delta
from rpy2.robjects import r

# TIMSS2007 可用维度合适
# data = np.array(pd.read_csv('./data/TIMSS/TIMSS2007/data.csv', header=None))
# Q = np.array(pd.read_excel('./data/TIMSS/TIMSS2007/q_m.xlsx', header=None))
# print("数据集为TIMSS2007")
# data_name = 'TIMSS2007'

# math/FrcSub  分数计算 拟合度较差  delatorre用过的数据，可用维度很小
data = np.array(pd.read_csv('data/FrcSub/FrcSub1/data.csv', header=None))
Q = np.array(pd.read_excel('./data/math2015/FrcSub1/q_m.xlsx', header=None))
Q[4,0] = 1
print("数据集为math2015/FrcSub1")
data_name = 'math2015/FrcSub1'

# 知识点维数太多，不可用
# data/FrcSub/FrcSub3 . Fraction Subtraction Data, 20 Items, 8 Hypothesized Skills
# The Q-matrix is from de la Torre and Douglas (2004)
# data = np.array(pd.read_csv('./data/FrcSub/FrcSub3/data.csv', header=None))
# Q = np.array(pd.read_excel('./data/FrcSub/FrcSub3/q_m.xlsx', header=None))
# print("数据集为FrcSub/FrcSub3")
# data_name = 'FrcSub/FrcSub3'

# TIMSS2003 # 维度太高，时间长验证费时
# data = np.array(pd.read_csv('./data/TIMSS/TIMSS2003/data.csv', header=None))
# Q = np.array(pd.read_excel('./data/TIMSS/TIMSS2003/q_m.xlsx', header=None))
# print("数据集为TIMSS2003")
# data_name = 'TIMSS2003'

# pisa 2000 德国子数据集导入
# data = np.array(pd.read_excel('./data/PISA 2000/data.xlsx', sheet_name='Sheet1', header=None))
# Q = np.array(pd.read_excel('./data/PISA 2000/q_m.xlsx', sheet_name='Sheet1', header=None))
# print("数据集为PISA 2000")
# data_name = 'PISA 2000'

# 读取math2015/Math1/data.xlsx  delatorre使用的数据
# data = np.array(pd.read_excel('./data/math2015/Math1/data.xlsx', sheet_name='Sheet1', header=None))
# Q = np.array(pd.read_csv('./data/math2015/Math1/q_m.csv', header=None))
# Q = Q[:15, :]
# print("数据集为math2015/Math1")
# data_name = 'math2015/Math1'


# ============================================================
students_num = data.shape[0]
items_num = Q.shape[0]
skills_num = Q.shape[1]

# 直接使用DINA模型结果
cdm_origin = DINA(data, Q, students_num, items_num, skills_num, skip_value=-1)
cdm_origin.train(epoch=2, epsilon=1e-3)
wrong_Q = Q.copy()
#  =============================== 使用假设检验方法修正Q矩阵
t1 = time.time()
alpha = 0.05
ht = Hypothetical_skill(wrong_Q, data, students_num, items_num, skills_num, alpha=alpha)
modify_q = ht.modify_Qj_method3_loop(0, 0.05)
print("alpha:", alpha)
# modify_q = ht.modify_Q_method3(mode='loop')  # 二次修改没有变化是因为待修改矩阵还是wrong_Q
# cdm_ht = DINA(data, modify_q, students_num, items_num, skills_num, skip_value=-1)
# cdm_ht.train(epoch=2, epsilon=1e-3)
t2 = time.time()
print("ht time cost:", t2 - t1)
#  ================================= delta法
epsilon = 0.01
delta = Delta(wrong_Q, data, students_num, items_num, skills_num, epsilon=epsilon, mode='dependence')
modify_q_delta = delta.modify_Q()
print("epsilon:", epsilon)
# modify_q_delta = delta.modify_qvector(item=2,epsilon=0.05)
# cdm_delta = DINA(data, modify_q_delta, students_num, items_num, skills_num, skip_value=-1)
# cdm_delta.train(epoch=2, epsilon=1e-3)
t3 = time.time()
print("delta time cost:", t3 - t2)
# ============================ gamma法
from code_functions.model.gamma import Gamma

gamma = Gamma(wrong_Q, data, students_num, items_num, skills_num, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
modify_q_gamma = gamma.modify_Q()
# modify_q_gamma = gamma.modify_qvector(item=2)
# cdm_gamma = DINA(data, modify_q_gamma, students_num, items_num, skills_num, skip_value=-1)
# cdm_gamma.train(epoch=2, epsilon=1e-3)
t4 = time.time()
print("gamma time cost:", t4 - t3)

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects

# print(r('.libPaths()'))
# 初始化pandas接口
pandas2ri.activate()
# 将Pandas DataFrame转换为R数据框
dat = pandas2ri.py2rpy(pd.DataFrame(data))

r("library(GDINA)")
if data_name == 'TIMSS2003':
    # modify_q_delta_delete = np.delete(modify_q_delta, [5, 10], axis=1)
    # modify_q_gamma_delete = np.delete(modify_q_gamma, [5, 10], axis=1)
    modify_q_delete = np.delete(modify_q, [5, 10], axis=1)
    Q_delete = np.delete(Q, [5, 10], axis=1)
    robjects.globalenv['Q_wrong'] = pandas2ri.py2rpy(pd.DataFrame(Q_delete))
    robjects.globalenv['dat_r'] = pandas2ri.py2rpy(pd.DataFrame(dat))
    robjects.globalenv['Q_ht'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delete))
    # robjects.globalenv['Q_delta'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delta_delete))
    # robjects.globalenv['Q_gamma'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_gamma_delete))
elif data_name == 'FrcSub/FrcSub3':
    # modify_q_delta_delete = np.delete(modify_q_delta, [5, ], axis=1)
    # modify_q_gamma_delete = np.delete(modify_q_gamma, [5], axis=1)
    modify_q_delete = np.delete(modify_q, [5], axis=1)
    Q_delete = np.delete(Q, [5], axis=1)
    robjects.globalenv['Q_wrong'] = pandas2ri.py2rpy(pd.DataFrame(Q_delete))
    robjects.globalenv['dat_r'] = pandas2ri.py2rpy(pd.DataFrame(dat))
    robjects.globalenv['Q_ht'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delete))
    # robjects.globalenv['Q_delta'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delta_delete))
    # robjects.globalenv['Q_gamma'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_gamma_delete))
else:
    robjects.globalenv['Q_wrong'] = pandas2ri.py2rpy(pd.DataFrame(Q))
    robjects.globalenv['dat_r'] = pandas2ri.py2rpy(pd.DataFrame(dat))
    robjects.globalenv['Q_ht'] = pandas2ri.py2rpy(pd.DataFrame(modify_q))
    robjects.globalenv['Q_delta'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delta))
    robjects.globalenv['Q_gamma'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_gamma))

# 转置
r("dat = t(dat_r)")
# DINA
fit_t1 = time.time()
r("est.Q_wrong = GDINA(dat,Q_wrong,model='DINA',verbose = 0)")
r("est.Q_ht = GDINA(dat,Q_ht,model='DINA',verbose = 0)")
r("est.Q_delta = GDINA(dat,Q_delta,model='DINA',verbose = 0)")
r("est.Q_gamma = GDINA(dat,Q_gamma,model='DINA',verbose = 0)")
r("a = anova(est.Q_wrong,est.Q_ht,est.Q_delta,est.Q_gamma)")
fit_t2 = time.time()
print("fit time cost:", fit_t2 - fit_t1)
print(r("a"))

# # 绝对拟合度
print(r("modelfit(est.Q_wrong)"))
print("======================================================")
print(r("modelfit(est.Q_ht)"))
print("======================================================")
print(r("modelfit(est.Q_delta)"))
print("======================================================")
print(r("modelfit(est.Q_gamma)"))

# 保存modify_q
# modify_q = pd.DataFrame(modify_q)
# modify_q.to_csv('./report/result/modify_q_ht.csv', index=False)
# Q = pd.DataFrame(Q)
# Q.to_csv('./report/result/Q.csv', index=False)
# modify_q_delta = pd.DataFrame(modify_q_delta)
# modify_q_delta.to_csv('./report/result/modify_q_delta.csv', index=False)
# modify_q_gamma = pd.DataFrame(modify_q_gamma)
# modify_q_gamma.to_csv('./report/result/modify_q_gamma.csv', index=False)

# ====================================== R语言的例子用GDINA  ===================================================
# print(a)
# r("a$'Pr(>Chi)'")
# print(a)
# r("Q1 = data.matrix(Q1)")
# r("")
#
# rscript = """
# library(GDINA)
# dat <- data.matrix(dat)
# Q_wrong <- data.matrix(Q_wrong)
# Q1 <- data.matrix(Q1)
# est.Q_wrong <- GDINA(dat,Q_wrong,model="DINA",verbose = 0)
# # est.Q_ht <- GDINA(dat,Q1,model="DINA",verbose = 0)
# """
# print(r(rscript))
#
# rscript = """
# library(GDINA)
#
# dat <- sim10GDINA$simdat
# Q <- matrix(c(1,0,0,
#               0,1,0,
#               0,0,1,
#               1,0,1,
#               0,1,1,
#               1,1,0,
#               1,0,1,
#               1,1,0,
#               1,1,1,
#               1,0,1),byrow = T,ncol = 3)
#
# est.Q_wrong <- GDINA(dat,Q,model="DINA",verbose = 0)
# est.Q_ht <- GDINA(dat,Q1,model="DINA",verbose = 0)
# # est.Q_delta <- GDINA(dat,Q_delta,model="DINA",verbose = 0)
# # est.Q_gamma <- GDINA(dat,Q_gamma,model="DINA",verbose = 0)
# # anova(est.Q_wrong,est.Q_ht,est.Q_delta,est.Q_gamma)
# anova(est.Q_wrong,est.Q_ht)
# """
# print(r(rscript))

# est.wald <- GDINA(dat, sugQ, model = extract(mc,"selected.model")$models, verbose = 0)
# anova(est.sugQ,est.wald)


# ============================================ 自己编写的拟合指标 ================================================
# 计算模型负2log似然值
# def neg_log_likelihood(cdm):
#     p = 0
#     for student in range(cdm.R.shape[0]):
#         for item in range(cdm.R.shape[1]):
#             g = cdm.guess[item]
#             s = cdm.slip[item]
#             # student学生掌握情况
#             state = cdm.all_states[cdm.theta[student], :]
#             # item题目知识点考察情况
#             q = cdm.q_m[item, :]
#             eta = 1
#             for k in range(cdm.know_num):
#                 eta *= state[k]**q[k]
#             # if np.all(np.array(state) >= np.array(q)):
#             #     eta = 1
#             # else:
#             #     eta = 0
#             x = cdm.R[student, item]
#             p += np.log(
#                 ((1 - s) ** x * s ** (1 - x)) ** eta * (g ** x * (1 - g) ** (1 - x)) ** (1 - eta)
#             )
#     return -2 * p


# def AIC(cdm):
#     # guess和
#     return 2*(len(cdm.guess)+len(cdm.slip)+cdm.state_num) +neg_log_likelihood(cdm)
# def BIC(cdm):
#     return len(cdm.guess)+len(cdm.slip)+cdm.state_num*np.log(cdm.R.shape[0]) + neg_log_likelihood(cdm)


# neg_log_likelihood(cdm_origin)
# neg_log_likelihood(cdm_ht)
# neg_log_likelihood(cdm_delta)
# neg_log_likelihood(cdm_gamma)

# AIC(cdm_origin)
# AIC(cdm_ht)
# AIC(cdm_delta)
# AIC(cdm_gamma)

# BIC(cdm_origin)
# BIC(cdm_ht)
# BIC(cdm_delta)
# BIC(cdm_gamma)


# 计算Q到modify_y，0改成1 的数量
# print(np.sum(Q != modify_q))
# print(np.sum(Q != modify_q_delta))
# print(np.sum(Q != modify_q_gamma))
