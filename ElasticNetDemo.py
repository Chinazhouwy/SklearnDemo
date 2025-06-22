import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import talib as ta
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
STOCK_CODE = "001222.SZ"  # 源飞宠物股票代码
DATA_DIR = Path("stock_data")  # 本地数据存储目录
CACHE_DAYS = 3  # 本地缓存有效天数（超过该天数自动更新）


def get_cache_path(code, start_date, end_date):
    """生成本地缓存文件路径"""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{code}_{start_date}_{end_date}.csv"


def check_cache_valid(cache_path, end_date):
    """检查缓存是否有效：文件存在且未过期"""
    if not cache_path.exists():
        return False

    # 检查文件修改时间是否在有效期内
    last_modified = datetime.fromtimestamp(cache_path.stat().st_mtime)
    current_date = datetime.today()
    if (current_date - last_modified).days > CACHE_DAYS:
        return False

    # 检查缓存数据是否覆盖目标日期范围
    try:
        df_cache = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
        return df_cache.index.min().strftime('%Y-%m-%d') <= start_date and \
            df_cache.index.max().strftime('%Y-%m-%d') >= end_date
    except:
        return False


def get_stock_data(code, start_date, end_date):
    """带本地缓存的股票数据获取函数"""
    cache_path = get_cache_path(code, start_date, end_date)

    # 优先使用有效缓存
    if check_cache_valid(cache_path, end_date):
        print(f"使用本地缓存数据: {cache_path}")
        return pd.read_csv(cache_path, parse_dates=['date'], index_col='date')

    # 缓存无效时调用远程接口
    print("本地缓存无效/缺失，调用baostock获取数据...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock登录失败，错误码：{lg.error_code}，错误信息：{lg.error_msg}")
        return None

    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 后复权
    )

    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if not data_list:
        print("未获取到股票数据")
        return None

    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.astype(float)

    # 保存到本地缓存
    df.to_csv(cache_path)
    print(f"数据已保存至本地缓存: {cache_path}")
    return df


def add_technical_indicators(df):
    """添加技术指标（与原逻辑一致）"""
    df_feat = df.copy()

    # 移动平均线
    df_feat['ma5'] = ta.MA(df_feat['close'], timeperiod=5)
    df_feat['ma10'] = ta.MA(df_feat['close'], timeperiod=10)
    df_feat['ma20'] = ta.MA(df_feat['close'], timeperiod=20)
    df_feat['ma60'] = ta.MA(df_feat['close'], timeperiod=60)

    # RSI指标
    df_feat['rsi14'] = ta.RSI(df_feat['close'], timeperiod=14)

    # MACD指标
    macd, signal, hist = ta.MACD(df_feat['close'])
    df_feat['macd'] = macd
    df_feat['macd_signal'] = signal
    df_feat['macd_hist'] = hist

    # KDJ指标
    slowk, slowd = ta.STOCH(df_feat['high'], df_feat['low'], df_feat['close'])
    df_feat['kdj_k'] = slowk
    df_feat['kdj_d'] = slowd
    df_feat['kdj_j'] = 3 * slowk - 2 * slowd

    # 布林带
    upper, middle, lower = ta.BBANDS(df_feat['close'], timeperiod=20)
    df_feat['bb_upper'] = upper
    df_feat['bb_middle'] = middle
    df_feat['bb_lower'] = lower
    df_feat['bb_width'] = (upper - lower) / middle

    # 成交量指标
    df_feat['volume_ma5'] = ta.MA(df_feat['volume'], timeperiod=5)
    df_feat['volume_ma10'] = ta.MA(df_feat['volume'], timeperiod=10)

    # 目标变量：下一日涨跌幅
    df_feat['next_return'] = df_feat['close'].pct_change(-1) * 100
    df_feat = df_feat.dropna()

    return df_feat


def train_elastic_net_model(df_feat):
    """模型训练（与原逻辑一致）"""
    X = df_feat.drop(['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'next_return'], axis=1)
    y = df_feat['next_return']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )

    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"训练集R²: {best_model.score(X_train, y_train):.4f}")
    print(f"测试集R²: {best_model.score(X_test, y_test):.4f}")

    return best_model, scaler, X, y, X_test, y_test


def visualize_results(y_test, y_pred, features):
    """可视化结果（精简版）"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='实际涨跌幅')
    plt.plot(y_pred, label='预测涨跌幅', alpha=0.7)
    plt.title('涨跌幅预测与实际值对比')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    importance = pd.Series(np.abs(model.coef_), index=features.columns)
    importance[importance > 0].sort_values().plot(kind='barh')
    plt.title('特征重要性（非零系数）')
    plt.show()


if __name__ == "__main__":
    start_date = "2022-06-22"
    end_date = "2025-06-21"

    # 获取数据（优先本地缓存）
    df = get_stock_data(STOCK_CODE, start_date, end_date)
    if df is None:
        exit()

    # 特征工程与模型训练
    df_feat = add_technical_indicators(df)
    model, scaler, X, y, X_test, y_test = train_elastic_net_model(df_feat)

    # 预测与可视化
    y_pred = model.predict(scaler.transform(X_test))
    visualize_results(y_test, y_pred, X)
