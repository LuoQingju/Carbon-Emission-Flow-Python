# Python复现论文《电力系统碳排放流的计算方法初探》[1]。
# [1] 周天睿,康重庆,徐乾耀,等.电力系统碳排放流的计算方法初探[J].电力系统自动化,2012,36(11):44-49.
#
# 环境：numpy 1.26.4、scipy 1.13.0、pypower 5.1.16
#
# 作者：罗清局
# 邮箱：luoqingju@qq.com
#
# 作者水平有限，难免会有错误及不足之处，敬请批评指正！

from numpy import array, shape, arange, ones, zeros

from scipy.sparse import csc_matrix, vstack
from scipy.sparse.linalg import spsolve

from pypower.api import case14, ppoption, rundcpf
from pypower.idx_bus import PD
from pypower.idx_gen import GEN_BUS, PG
from pypower.idx_brch import F_BUS, T_BUS, PF, PT

ppc = case14()

Pg = array([120, 40, 60, 19, 20])  # 设置发电机有功出力

ppc['gen'][:, PG] = Pg

ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
res = rundcpf(ppc, ppopt)  # MATPOWER：计算直流潮流

if res[0]['success'] != 1:
    print('----------直流潮流计算失败！----------')
    exit()

N = shape(res[0]['bus'])[0]    # 节点数（母线数）
K = shape(res[0]['gen'])[0]    # 发电机数

Pd = res[0]['bus'][:, PD]  # 节点负荷
gen_bus = res[0]['gen'][:, GEN_BUS]  # 发电机节点
gen_bus = gen_bus.astype(int) - 1

fbus = res[0]['branch'][:, F_BUS]  # 线路 "from" 端节点
tbus = res[0]['branch'][:, T_BUS]  # 线路 "to" 端节点
fbus = fbus.astype(int) - 1
tbus = tbus.astype(int) - 1

Pl_from = res[0]['branch'][:, PF]  # 线路 "from" 端功率
Pl_to = res[0]['branch'][:, PT]  # 线路 "to" 端功率

Pl_from[Pl_from < 0] = 0  # 反向的功率置零
Pl_to[Pl_to < 0] = 0  # 反向的功率置零

idx_PF = Pl_from > 0  # 线路 "from" 端功率索引
PB_F_Mat = csc_matrix((Pl_from[idx_PF], (fbus[idx_PF], tbus[idx_PF])), (N, N))  # 线路 "from" 端潮流分布矩阵

idx_PT = Pl_to > 0  # 线路 "to" 端功率索引
PB_T_Mat = csc_matrix((Pl_to[idx_PT], (tbus[idx_PT], fbus[idx_PT])), (N, N))  # 线路 "to" 端潮流分布矩阵

# 支路潮流分布矩阵(branch power flow distribution matrix) N 阶方阵
PB_Mat = PB_F_Mat + PB_T_Mat

# 机组注入分布矩阵(power injection distribution matrix) K×N 阶矩阵
PG_Mat = csc_matrix((Pg, (arange(K), gen_bus)), (K, N))

# 负荷分布矩阵(load distribution matrix) M×N 阶矩阵 M 为负荷数
# 为了简化，假设每个节点都存在负荷，不存在负荷的节点按照负荷为零处理
# 所以，负荷分布矩阵变为 N×N 阶矩阵
PL_Mat = csc_matrix((Pd, (arange(N), arange(N))), (N, N))

PZ_Mat = vstack([PB_Mat, PG_Mat], 'csc')

# 节点有功通量矩阵(nodal active power flux matrix) N 阶对角阵
PN_Mat = csc_matrix((ones(N+K)@PZ_Mat, (arange(N), arange(N))))  # 论文中的公式2

# 发电机组碳排放强度向量(unit carbon emission intensity vector)
EG_Vec = array([875, 525, 0, 520, 0])  # 论文中的公式14

# 节点碳势向量(nodal carbon intensity vector)
EN_Vec = spsolve(PN_Mat - PB_Mat.T, PG_Mat.T @ EG_Vec)  # 论文中的公式13

# 支路碳流率分布矩阵 (branch carbon emission flow rate distribution matrix) N 阶方阵
# 论文中的公式5可能存在笔误，应该是 RB = diag(EN)*PB
EN_Mat = csc_matrix((EN_Vec, (arange(N), arange(N))), (N, N))
RB_Mat = EN_Mat.dot(PB_Mat)

RB_Mat = RB_Mat/1000  # kgCO2/h ==> tCO2/h

# 负荷碳流率向量(load carbon emission rate vector)
RL_Vec = PL_Mat @ EN_Vec  # 论文中的公式7
RL_Vec = RL_Vec/1000  # kgCO2/h ==> tCO2/h

# 支路碳流密度(branch carbon emission flow intensity)
spones_PB_Mat = csc_matrix((ones(len(RB_Mat.data)), RB_Mat.indices, RB_Mat.indptr), (N, N))
EB_Mat = EN_Mat.dot(spones_PB_Mat)

# 机组注入碳流率
IN_Vec = PG_Mat.T @ EG_Vec
IN_Vec = IN_Vec/1000  # kgCO2/h ==> tCO2/h

tol = 1e-5  # 判断输出是否为0的阈值
print('\n', end='')

# 表2 节点有功通量与节点碳势
print('表2 节点有功通量与节点碳势')
print('----------------------------------------')
print('节点    节点有功通量/MW    节点碳势(gCO2/kWh)')
print('----------------------------------------')
for i in range(N):
    print('%2d' % (i + 1), end='')
    if abs(PN_Mat[i, i]) < tol:
        print('%13d   ' % 0, end='')
    else:
        print('%16.2f' % PN_Mat[i, i], end='')
    if abs(EN_Vec[i]) < tol:
        print('%13d   ' % 0, end='')
    else:
        print('%16.2f' % EN_Vec[i], end='')
    print('\n', end='')
print('----------------------------------------')
print('\n', end='')

# 表3 支路有功潮流与碳流率
L = shape(res[0]['branch'])[0]  # 线路数
Pl = Pl_from - Pl_to  # 线路功率
EB_Vec = zeros(L)  # 支路碳流密度
RB_Vec = zeros(L)  # 支路碳流率
for i in range(L):
    if Pl[i] > 0:
        EB_Vec[i] = EB_Mat[fbus[i], tbus[i]]
        RB_Vec[i] = RB_Mat[fbus[i], tbus[i]]
    else:
        EB_Vec[i] = EB_Mat[tbus[i], fbus[i]]
        RB_Vec[i] = RB_Mat[tbus[i], fbus[i]]

print('表3 支路有功潮流与碳流率')
print('---------------------------------------------------------------------------')
print('起始节点    终止节点    支路有功潮流(MW)    支路碳流密度(gCO2/kWh)    碳流率(tCO2/h)')
for i in range(L):
    print('%4d %9d' % (fbus[i] + 1, tbus[i] + 1), end='')
    if abs(Pl[i]) < tol:
        print('%13d   ' % 0, end='')
    else:
        print('%16.2f' % Pl[i], end='')
    if abs(EB_Vec[i]) < tol:
        print('%16d   ' % 0, end='')
    else:
        print('%19.2f' % EB_Vec[i].astype(float), end='')
    if abs(RB_Vec[i]) < tol:
        print('%18d   ' % 0, end='')
    else:
        print('%21.2f' % RB_Vec[i].astype(float), end='')
    print('\n', end='')
print('---------------------------------------------------------------------------')
print('\n', end='')

# 表4 负荷碳流率和机组注入碳流率
print('表4 负荷碳流率和机组注入碳流率')
print('-----------------------------------------------')
print('节点    负荷碳流率(tCO2/h)    机组注入碳流率(tCO2/h)')
for i in range(N):
    print('%2d' % (i + 1), end='')
    if abs(RL_Vec[i]) < tol:
        print('%11d   ' % 0, end='')
    else:
        print('%14.2f' % RL_Vec[i], end='')
    if abs(IN_Vec[i]) < tol:
        print('%18d   ' % 0, end='')
    else:
        print('%21.2f' % IN_Vec[i].astype(float), end='')
    print('\n', end='')
print('-----------------------------------------------')
print('\n', end='')
