""" 研究三 实例验证 调用R代码计算拟合度 """
import time

import numpy as np
import pandas as pd
from code_functions.EduCDM import EMDINA as DINA
from code_functions.model.hypothesis_skill import Hypothetical_skill
from code_functions.model.delta import Delta
from rpy2.robjects import r
from code_functions.model.gamma import Gamma
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects

r("library(GDINA)")
# ======================================= 数据导入
# # TIMSS2007
data = np.array(pd.read_csv('./data/TIMSS/TIMSS2007/data.csv', header=None))
Q = np.array(pd.read_excel('./data/TIMSS/TIMSS2007/q_m.xlsx', header=None))
print("数据集为TIMSS2007")
data_name = 'TIMSS2007'

# math/FrcSub  分数计算 拟合度较差  delatorre用过的数据
data = np.array(pd.read_csv('data/FrcSub/FrcSub1/data.csv', header=None))
Q = np.array(pd.read_excel('./data/math2015/FrcSub1/q_m.xlsx', header=None))
Q[4, 0] = 1
print("数据集为math2015/FrcSub1")
data_name = 'FrcSub1'

# 参数设置
students_num = data.shape[0]
items_num = Q.shape[0]
skills_num = Q.shape[1]
item = 0  # 第item道题目
result_pd = pd.DataFrame()
for item in range(0, items_num):
    print(f"第{item}道题目")
    result = []
    # ======================================= object1:DINA
    cdm_origin = DINA(data, Q, students_num, items_num, skills_num, skip_value=-1)
    cdm_origin.train(epoch=2, epsilon=1e-3)
    wrong_Q = Q.copy()
    # ===============================  object2:使用假设检验方法修正Q矩阵
    t1 = time.time()
    alpha = 0.05
    ht = Hypothetical_skill(wrong_Q, data, students_num, items_num, skills_num, alpha=alpha)
    modify_q = ht.modify_Qj_method3_loop(item, alpha=alpha)
    t2 = time.time()
    print("alpha:", alpha)
    print("ht time cost:", t2 - t1)
    #  ================================= object3:delta法修正Q矩阵
    epsilon = 0.01
    delta = Delta(wrong_Q, data, students_num, items_num, skills_num, epsilon=epsilon, mode='dependence')
    modify_q_delta = delta.modify_qvector(item=item, epsilon=epsilon)
    t3 = time.time()
    print("epsilon:", epsilon)
    print("delta time cost:", t3 - t2)
    # ============================ object4:gamma法修正Q矩阵
    gamma = Gamma(wrong_Q, data, students_num, items_num, skills_num, threshold_g=0.2, threshold_s=0.2,
                  threshold_es=0.2)
    modify_q_gamma = gamma.modify_qvector(item=item)
    t4 = time.time()
    print("gamma time cost:", t4 - t3)

    # print(r('.libPaths()')) # 查看R包路径
    pandas2ri.activate()  # 初始化pandas接口
    # ======================================= 将Pandas DataFrame转换为R数据框
    robjects.globalenv['Q_wrong'] = pandas2ri.py2rpy(pd.DataFrame(Q))
    robjects.globalenv['dat'] = pandas2ri.py2rpy(pd.DataFrame(data))
    robjects.globalenv['Q_ht'] = pandas2ri.py2rpy(pd.DataFrame(modify_q))
    robjects.globalenv['Q_delta'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_delta))
    robjects.globalenv['Q_gamma'] = pandas2ri.py2rpy(pd.DataFrame(modify_q_gamma))
    # ======================================= R代码计算拟合度
    try:
        r("est.Q_wrong = GDINA(dat,Q_wrong,model='DINA',verbose = 0)")
        r("est.Q_ht = GDINA(dat,Q_ht,model='DINA',verbose = 0)")
        r("est.Q_delta = GDINA(dat,Q_delta,model='DINA',verbose = 0)")
        r("est.Q_gamma = GDINA(dat,Q_gamma,model='DINA',verbose = 0)")
        r("a = anova(est.Q_wrong,est.Q_ht,est.Q_delta,est.Q_gamma)")
        anova = r("anova(est.Q_wrong,est.Q_ht,est.Q_delta,est.Q_gamma)")
    # 捕捉错误
    except Exception as e:
        print(e)
        continue
    if data_name =="FrcSub1":
        fit_wrongQ = r("modelfit(est.Q_wrong)")
        fit_ht = r("modelfit(est.Q_ht)")
        fit_delta = r("modelfit(est.Q_delta)")
        fit_gamma = r("modelfit(est.Q_gamma)")
        # print(r("modelfit(est.Q_ht)"))
        # # 绝对拟合度
        # 假设检验相对original的差值
        delta_ht_2ll = eval(anova[0][2][1]) - eval(anova[0][2][0])
        delta_ht_aic = eval(anova[0][3][1]) - eval(anova[0][3][0])
        delta_ht_bic = eval(anova[0][4][1]) - eval(anova[0][4][0])
        delta_ht_m2 = fit_ht[0][0] - fit_wrongQ[0][0]
        delta_ht_rmsea = fit_ht[4][0] - fit_wrongQ[4][0]
        delta_ht_srmsr = fit_ht[3][0] - fit_wrongQ[3][0]
        delta_delta_2ll = eval(anova[0][2][2]) - eval(anova[0][2][0])
        delta_delta_aic = eval(anova[0][3][2]) - eval(anova[0][3][0])
        delta_delta_bic = eval(anova[0][4][2]) - eval(anova[0][4][0])
        delta_delta_m2 = fit_delta[0][0] -fit_wrongQ[0][0]
        delta_delta_rmsea = fit_delta[4][0] - fit_wrongQ[4][0]
        delta_delta_srmsr = fit_delta[3][0] - fit_wrongQ[3][0]
        delta_gamma_2ll = eval(anova[0][2][3]) - eval(anova[0][2][0])
        delta_gamma_aic = eval(anova[0][3][3]) - eval(anova[0][3][0])
        delta_gamma_bic = eval(anova[0][4][3]) - eval(anova[0][4][0])
        delta_gamma_m2 = fit_gamma[0][0] - fit_wrongQ[0][0]
        delta_gamma_rmsea = fit_gamma[4][0] - fit_wrongQ[4][0]
        delta_gamma_srmsr = fit_gamma[3][0] - fit_wrongQ[3][0]
        result.append([eval(anova[0][2][0]), eval(anova[0][3][0]), eval(anova[0][4][0]),
                       fit_wrongQ[0][0], fit_wrongQ[4][0], fit_wrongQ[3][0]])
        result.append([delta_ht_2ll, delta_ht_aic, delta_ht_bic, delta_ht_m2, delta_ht_rmsea, delta_ht_srmsr])
        result.append([delta_delta_2ll, delta_delta_aic, delta_delta_bic, delta_delta_m2, delta_delta_rmsea, delta_delta_srmsr])
        result.append([delta_gamma_2ll, delta_gamma_aic, delta_gamma_bic, delta_gamma_m2, delta_gamma_rmsea, delta_gamma_srmsr])
    # 原本的数据，不进行差值
    # result.append([eval(anova[0][2][0]), eval(anova[0][3][0]), eval(anova[0][4][0]),
    #                fit_wrongQ[0][0], fit_wrongQ[2][0],fit_wrongQ[1][0],fit_wrongQ[4][0], fit_wrongQ[3][0]])
    # result.append([eval(anova[0][2][1]), eval(anova[0][3][1]), eval(anova[0][4][1]),
    #                fit_ht[0][0], fit_ht[2][0],fit_ht[1][0],fit_ht[4][0],fit_ht[3][0]])
    # result.append([eval(anova[0][2][2]), eval(anova[0][3][2]), eval(anova[0][4][2]),
    #                fit_delta[0][0],fit_delta[2][0],fit_delta[1][0],fit_delta[4][0], fit_delta[3][0]])
    # result.append([eval(anova[0][2][3]), eval(anova[0][3][3]), eval(anova[0][4][3]),
    #                fit_gamma[0][0], fit_gamma[2][0],fit_gamma[1][0],fit_gamma[4][0], fit_gamma[3][0]])
    # pd.DataFrame(result)
        result_pd = pd.concat([result_pd, pd.DataFrame(result)], axis=0)
    elif data_name == "TIMSS2007":
        # 假设检验相对original的差值
        delta_ht_2ll = eval(anova[0][2][1]) - eval(anova[0][2][0])
        delta_ht_aic = eval(anova[0][3][1]) - eval(anova[0][3][0])
        delta_ht_bic = eval(anova[0][4][1]) - eval(anova[0][4][0])

        delta_delta_2ll = eval(anova[0][2][2]) - eval(anova[0][2][0])
        delta_delta_aic = eval(anova[0][3][2]) - eval(anova[0][3][0])
        delta_delta_bic = eval(anova[0][4][2]) - eval(anova[0][4][0])

        delta_gamma_2ll = eval(anova[0][2][3]) - eval(anova[0][2][0])
        delta_gamma_aic = eval(anova[0][3][3]) - eval(anova[0][3][0])
        delta_gamma_bic = eval(anova[0][4][3]) - eval(anova[0][4][0])

        result.append([eval(anova[0][2][0]), eval(anova[0][3][0]), eval(anova[0][4][0])])
        result.append([delta_ht_2ll, delta_ht_aic, delta_ht_bic])
        result.append([delta_delta_2ll, delta_delta_aic, delta_delta_bic])
        result.append([delta_gamma_2ll, delta_gamma_aic, delta_gamma_bic])

        result_pd = pd.concat([result_pd, pd.DataFrame(result)], axis=0)

# 保存结果
result_pd.to_csv(f'report/result/research3/{data_name}_each_item_result.csv', index=False)
