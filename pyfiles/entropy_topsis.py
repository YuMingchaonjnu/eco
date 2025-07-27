import pandas as pd
import numpy as np

def data_normalization(df, pos_var=None, neg_var=None):
    """
    对DataFrame中的变量进行归一化处理。
    
    参数：
    ----------
    df : pandas.DataFrame
        原始数据，需为数值型变量。
    pos_var : list[str], optional
        正向指标列名列表（值越大越好）。
    neg_var : list[str], optional
        负向指标列名列表（值越小越好）。

    返回：
    ----------
    df_norm : pandas.DataFrame
        归一化后的数据框，各列数据值归一到 [0, 1]。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 df 必须是一个 pandas.DataFrame")

    df = df.copy()
    all_columns = df.columns.tolist()

    # 如果正负变量未指定，默认全部按正向处理
    if pos_var is None and neg_var is None:
        pos_var = all_columns
        neg_var = []
    else:
        pos_var = pos_var or []
        neg_var = neg_var or []

    # 校验列名合法性
    invalid_cols = set(pos_var + neg_var) - set(all_columns)
    if invalid_cols:
        raise ValueError(f"下列列名在 DataFrame 中未找到: {invalid_cols}")

    df_norm = pd.DataFrame(index=df.index)

    # 正向指标归一化（值越大越好）
    for col in pos_var:
        min_val, max_val = df[col].min(), df[col].max()
        range_val = max_val - min_val
        df_norm[col] = (df[col] - min_val) / range_val if range_val != 0 else 0.0

    # 负向指标归一化（值越小越好）
    for col in neg_var:
        min_val, max_val = df[col].min(), df[col].max()
        range_val = max_val - min_val
        df_norm[col] = (max_val - df[col]) / range_val if range_val != 0 else 0.0

    # 保持原列顺序
    df_norm = df_norm[all_columns]

    return df_norm

def entropy_weights(df_norm: pd.DataFrame, epsilon: float = 1e-12) -> np.ndarray:
    """
    使用熵权法（Entropy Weight Method）计算各指标的权重。

    参数：
    ----------
    df_norm : pd.DataFrame
        已归一化的输入数据，每列代表一个评价指标，每行代表一个样本。
        数据应为非负数，并且按列归一或缩放到 [0,1] 区间。

    epsilon : float, 默认值为 1e-12
        避免 log(0) 或除以零的微小常数。

    返回：
    -------
    np.ndarray
        每个指标对应的熵权值数组。
    """
    if not isinstance(df_norm, pd.DataFrame):
        raise TypeError("输入必须为 pandas DataFrame 类型。")
        
    if (df_norm < 0).any().any():
        raise ValueError("归一化后的数据中不能包含负数。")

    m = df_norm.shape[0]  # 样本数量（行数）
    if m == 0:
        raise ValueError("输入的 DataFrame 至少应包含一行数据。")

    # 再次归一化，确保每列的总和为 1，形成概率分布
    p = df_norm / (df_norm.sum(axis=0) + epsilon)
    
    # 熵值计算公式
    k = 1 / np.log(m)
    e = -k * np.sum(p * np.log(p + epsilon), axis=0)

    # 计算冗余度（差异系数）并得出最终权重
    d = 1 - e
    w = d / np.sum(d)

    return w.values  # 返回 numpy 数组，便于后续数值处理


def topsis(df, weights, pos_var=None, neg_var=None):
    """
    基于TOPSIS方法的多指标决策函数。

    参数：
    ----------
    df : pd.DataFrame
        决策矩阵（行代表备选方案，列代表评价指标）。
    weights : array-like
        各指标的权重，应归一化或总和为1。
    pos_var : list 或 None
        正向（效益型）指标的名称列表。
    neg_var : list 或 None
        负向（成本型）指标的名称列表。

    返回：
    -------
    pd.Series
        每个备选方案相对于理想解的接近度得分（越大越优）。
    """

    # 1. 极差归一化
    R = (df - df.min()) / (df.max() - df.min())

    # 2. 加权归一化矩阵
    V = R * weights

    # 3. 理想解与负理想解的构造
    if pos_var is None or neg_var is None:
        # 如果未指定正向/负向指标，则所有指标都视为正向
        V_plus = V.max(axis=0)     # 最优解
        V_minus = V.min(axis=0)    # 最劣解
    else:
        cols = df.columns
        is_pos = np.isin(cols, pos_var)
        V_plus = np.where(is_pos, V.max(axis=0), V.min(axis=0))   # 正向取最大，负向取最小
        V_minus = np.where(is_pos, V.min(axis=0), V.max(axis=0))  # 正向取最小，负向取最大

    # 4. 计算与理想解/负理想解的距离
    S_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))   # 到最优解的距离
    S_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1)) # 到最劣解的距离

    # 5. 计算接近度系数（越大越好）
    C = S_minus / (S_plus + S_minus)

    return pd.Series(C, index=df.index)
