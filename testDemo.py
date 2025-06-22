import baostock as bs
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
STOCK_CODE = 'sh.600000'  # 浦发银行
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'
LOOK_BACK = 20  # 用过去20天数据构造特征
PRED_DAYS = 3   # 预测未来3天收益率是否为正

# ==================== 1. 数据获取 ====================
def get_stock_data():
    """获取前复权的历史日线数据"""
    lg = bs.login()
    if lg.error_code != '0':
        raise ValueError(f"baostock登录失败，错误码：{lg.error_code}")

    rs = bs.query_history_k_data_plus(
        STOCK_CODE,
        "date,open,high,low,close,volume,turn",
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="2"
    )
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 数据清洗与类型转换
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['open','high','low','close','volume','turn']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.dropna()  # 移除缺失值行
    
    bs.logout()
    return df

# ==================== 2. 特征工程 ====================
def create_features_labels(df):
    """构造技术指标特征和预测标签"""
    # 基础统计特征
    df['pct_change'] = df['close'].pct_change()  # 日涨跌幅
    df['volatility'] = df['pct_change'].rolling(LOOK_BACK).std()  # 波动率
    df['ma5'] = df['close'].rolling(5).mean() / df['close']  # 5日均值归一化
    df['ma20'] = df['close'].rolling(20).mean() / df['close']  # 20日均值归一化
    df['volume_ma5'] = df['volume'].rolling(5).mean() / df['volume']  # 成交量均线
    
    # 动量指标（RSI）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 构造标签：未来PRED_DAYS天收益率是否>0
    df['future_return'] = df['close'].pct_change(periods=PRED_DAYS).shift(-PRED_DAYS)
    df['label'] = (df['future_return'] > 0).astype(int)
    
    # 移除无效行（前20天+后3天）
    return df.dropna().reset_index(drop=True)

# ==================== 3. 数据预处理 ====================
def build_features(df):
    """构造时间窗口特征（机器学习需要二维特征矩阵）"""
    feature_cols = ['close','volume','turn','pct_change','volatility','ma5','ma20','rsi']
    X, y = [], []
    
    # 滑动窗口构造特征：每个样本包含过去LOOK_BACK天的特征
    for i in range(LOOK_BACK, len(df)-PRED_DAYS):
        window = df[feature_cols].iloc[i-LOOK_BACK:i].values  # 取过去20天数据
        X.append(window.flatten())  # 展平为一维特征向量（20天×8特征=160维）
        y.append(df['label'].iloc[i])  # 对应未来3天的标签
    
    return np.array(X), np.array(y)

# ==================== 4. 机器学习模型 ====================
def train_ml_model(X_train, y_train):
    """训练随机森林分类器（可替换为其他模型）"""
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )
    model.fit(X_train, y_train)
    return model

# ==================== 5. 回测与分析 ====================
def strategy_backtest(df, y_pred, test_indices):
    """简单策略回测：预测为1时持有，0时空仓"""
    # 提取测试期数据
    test_df = df.iloc[test_indices + LOOK_BACK:-PRED_DAYS].copy()  # 对齐特征窗口
    test_df['pred'] = y_pred
    test_df['strategy_return'] = test_df['pct_change'] * test_df['pred']  # 预测正确时获得收益
    
    # 计算累计收益
    test_df['baseline_cum'] = (1 + test_df['pct_change']).cumprod() - 1
    test_df['strategy_cum'] = (1 + test_df['strategy_return']).cumprod() - 1
    
    # 可视化
    plt.figure(figsize=(12,6))
    plt.plot(test_df['date'], test_df['baseline_cum'], label='基准收益（持有不动）')
    plt.plot(test_df['date'], test_df['strategy_cum'], label='策略收益')
    plt.title('机器学习策略回测结果')
    plt.xlabel('日期')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def feature_importance_analysis(model, feature_names):
    """特征重要性分析"""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(10,6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    plt.title("特征排列重要性")
    plt.xlabel("重要性得分")
    plt.tight_layout()
    plt.show()

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 数据获取与处理
    df = get_stock_data()
    df = create_features_labels(df)
    X, y = build_features(df)
    
    # 时间序列分割（避免未来数据泄漏）
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = next(tscv.split(X))  # 取最后一个分割作为训练/测试
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 模型训练与评估
    model = train_ml_model(X_train, y_train)
    y_pred = model.predict(X_test)
    print("模型评估报告：")
    print(classification_report(y_test, y_pred))
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2f}")
    
    # 特征重要性分析（构造特征名称）
    base_features = ['close','volume','turn','pct_change','volatility','ma5','ma20','rsi']
    feature_names = [f'day_{i}_{f}' for i in range(LOOK_BACK) for f in base_features]
    feature_importance_analysis(model, feature_names)
    
    # 策略回测
    strategy_backtest(df, y_pred, test_idx)
    