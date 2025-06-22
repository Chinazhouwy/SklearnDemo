import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

#
# Ridge（岭回归）和 `SGDRegressor(penalty='l2')` 虽然都引入了 L2 正则化，但在底层实现、优化方式和适用场景上有明显区别。以下是核心差异的详细说明：
#
#
# ### **1. 模型本质与优化目标相同，但求解方式不同**
# 两者的目标函数均为：
# $$ \text{minimize} \quad \frac{1}{2N}||Xw - y||^2_2 + \frac{\alpha}{2}||w||^2_2 $$
# （其中 \(N\) 是样本量，\(\alpha\) 是正则化强度）
# **数学本质一致**，但求解参数 \(w\) 的优化算法不同：
#
# - **Ridge**：通过闭式解（解析解）或批量优化算法（如 sag、cholesky 等）直接求解最优参数。
#   例如，当使用 `solver='sag'` 时，本质是批量梯度下降（或随机平均梯度），需要遍历所有样本计算梯度。
#
# - **SGDRegressor**：通过随机梯度下降（SGD）迭代优化。
#   每次仅用 1 个（或小批量）样本计算梯度并更新参数，**参数是逐步迭代逼近最优解**（而非直接求解闭式解）。
#
#
# ### **2. 计算效率与适用场景不同**
# | 特性                | Ridge                          | SGDRegressor                  |
# |---------------------|--------------------------------|--------------------------------|
# | **数据规模**         | 适合小/中规模数据（矩阵运算耗时随数据量增长） | 适合大规模数据（单次迭代仅处理少量样本） |
# | **内存占用**         | 需存储完整数据集（计算 \(X^TX\) 等矩阵） | 仅需存储当前批次数据（适合在线学习） |
# | **迭代灵活性**       | 一次性训练完成（无法增量更新） | 支持 `partial_fit` 在线学习（可逐步添加新数据） |
# | **超参数敏感性**     | 对正则化强度 \(\alpha\) 敏感（直接影响闭式解） | 对学习率、迭代次数更敏感（需调参避免震荡） |
#
#
# ### **3. 实现细节的差异**
# - **正则化范围**：
#   Ridge 的正则化项仅约束权重 \(w\)，截距项 \(b\) 不参与正则化（`fit_intercept=True` 时）；
#   SGDRegressor 的正则化默认同时约束 \(w\) 和 \(b\)（可通过 `fit_intercept=True` 控制是否对截距正则化）。
#
# - **优化停止条件**：
#   Ridge 通过求解器直接收敛（如 `sag` 会设置最大迭代次数）；
#   SGDRegressor 需显式设置 `max_iter`（最大迭代次数）和 `tol`（损失变化阈值），否则可能提前停止或过拟合。
#
#
# ### **代码验证差异（结合你的示例）**
# 你的代码中，Ridge 和 SGDRegressor 最终输出的系数（`coef_`）和截距（`intercept_`）可能接近，但受优化算法影响会有微小差异：
# - Ridge 的 `solver='sag'` 是批量优化，结果更稳定；
# - SGDRegressor 是随机梯度下降（受随机初始化和样本顺序影响），结果可能有随机波动（可通过设置 `random_state` 固定随机种子）。
#
#
# ### **总结：如何选择？**
# - 小数据/离线训练：优先选 Ridge（计算高效，结果稳定）；
# - 大数据/在线学习：选 SGDRegressor（内存友好，支持增量训练）；
# - 需动态更新模型（如实时数据流）：只能用 SGDRegressor。
#
#
# 如果需要验证两者的等价性，可以尝试将 SGDRegressor 的学习率、迭代次数调至极大（如 `max_iter=1e5`），并关闭随机扰动（`random_state=42`），最终参数会无限接近 Ridge 的结果。

if __name__ == '__main__':
    # 1、创建数据集X，y
    X = 2*np.random.rand(100, 5)
    w = np.random.randint(1,10,size = (5,1))
    b = np.random.randint(1,10,size = 1)
    y = X.dot(w) + b + np.random.randn(100, 1)

    print('原始方程的斜率：',w.ravel())
    print('原始方程的截距：',b)

    ridge = Ridge(alpha= 1, solver='sag')
    ridge.fit(X, y)
    print('岭回归求解的斜率：',ridge.coef_)
    print('岭回归求解的截距：',ridge.intercept_)

    # 线性回归梯度下降方法
    sgd = SGDRegressor(penalty='l2',alpha=0,l1_ratio=0)
    sgd.fit(X, y.reshape(-1,))
    print('随机梯度下降求解的斜率是：',sgd.coef_)
    print('随机梯度下降求解的截距是：',sgd.intercept_)


