from code_functions.model.hypothetical import is_refuse
import math
# 累计概率
n = 30
r = 8
p = 0.2
alpha = 0.05


cumulate_p = 0
for i in range(r+1, n+1):  # 注意Python的range不包括结束值，因此需要加1
    cumulate_p += math.comb(n, i) * (p ** i) * ((1-p) ** (n - i))

# np.isclose(cumulate_p, alpha, atol=1e-9) or cumulate_p < alpha else 0

# cumulate_p = 0
# for i in range(r, n + 1):  # 注意Python的range不包括结束值，因此需要加1
#     cumulate_p += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
# return 1 if np.isclose(cumulate_p, alpha, atol=1e-9) or cumulate_p < alpha else 0

i=20
math.comb(n, i) * (p ** i) * (p ** (n - i))