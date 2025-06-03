import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from jqdata import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深 300 作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式（真实价格）
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉 order 系列 API 产生的比 error 级别低的 log
    log.set_level('order', 'error')
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣 5 元
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    # 选股参数
    context.market_cap_threshold = 100  # 市值门槛（亿元）
    context.pe_ratio_threshold = 30  # 市盈率门槛
    # 择时参数
    context.short_window = 5  # 短期均线窗口
    context.long_window = 20  # 长期均线窗口
    # 机器学习参数
    context.lookback_days = 20  # 训练数据回溯天数
    # 风险控制参数
    context.stop_loss_ratio = 0.05  # 止损比例
    context.max_position = 0.8  # 最大仓位
    # 运行函数（reference_security 为运行时间的参考标的；传入的标的只做种类区分，因此传入 '000300.XSHG' 或 '510300.XSHG' 是一样的）
    # 每天开盘前选股
    run_daily(select_stocks, time='before_open', reference_security='000300.XSHG')
    # 每天交易时进行买卖决策
    run_daily(trade, time='every_bar', reference_security='000300.XSHG')

# 选股函数
def select_stocks(context):
    # 获取所有A股股票
    all_stocks = get_all_securities(types=['stock']).index.tolist()
    # 筛选市值大于门槛且市盈率小于门槛的股票
    q = query(valuation.code, valuation.market_cap, valuation.pe_ratio).filter(
        valuation.code.in_(all_stocks),
        valuation.market_cap > context.market_cap_threshold,
        valuation.pe_ratio < context.pe_ratio_threshold
    )
    df = get_fundamentals(q)
    context.selected_stocks = df['code'].tolist()

# 训练机器学习模型
def train_model(context, stock):
    # 获取历史数据
    history_data = attribute_history(stock, context.lookback_days, '1d', ['close'], skip_paused=True)
    if len(history_data) < context.lookback_days:
        return None
    # 计算特征和标签
    X = []
    y = []
    for i in range(1, len(history_data)):
        X.append([history_data['close'][i - 1]])
        y.append(1 if history_data['close'][i] > history_data['close'][i - 1] else 0)
    X = np.array(X)
    y = np.array(y)

    # 检查并处理无效值
    if np.isnan(X).any() or np.isinf(X).any():
        # 这里简单地删除包含无效值的样本，你也可以根据需求进行更复杂的处理，如插值
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

    # 确保处理后的数据足够用于训练
    if len(X) < 2:
        return None

    # 检查标签是否至少包含两个不同的类别
    if len(np.unique(y)) < 2:
        return None

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X, y)
    return model

# 交易函数
def trade(context):
    # 风险控制：检查持仓股票是否触发止损
    for stock in context.portfolio.positions.keys():
        position = context.portfolio.positions[stock]
        cost = position.avg_cost
        current_price = position.price
        if current_price <= cost * (1 - context.stop_loss_ratio):
            log.info('触发止损，卖出 %s' % stock)
            order_target(stock, 0)
    # 计算可用资金和最大可买入仓位
    available_cash = context.portfolio.available_cash
    max_buy_amount = context.portfolio.total_value * context.max_position - context.portfolio.positions_value
    if max_buy_amount < 0:
        max_buy_amount = 0
    # 遍历选股结果
    for stock in context.selected_stocks:
        # 训练机器学习模型
        model = train_model(context, stock)
        if model is None:
            continue
        # 获取最新收盘价
        last_close = history(1, '1d', 'close', stock)[stock][0]
        # 预测涨跌
        prediction = model.predict([[last_close]])[0]
        if prediction == 1:
            # 择时：短期均线上穿长期均线
            short_bars = get_bars(stock, context.short_window, '1d', 'close', include_now=True)
            long_bars = get_bars(stock, context.long_window, '1d', 'close', include_now=True)
            # 确保获取到足够的数据
            if short_bars is not None and len(short_bars) > 0 and long_bars is not None and len(long_bars) > 0:
                short_ma = np.mean(short_bars['close'])
                long_ma = np.mean(long_bars['close'])
                if short_ma > long_ma:
                    # 计算可买入数量
                    if available_cash > 0 and max_buy_amount > 0:
                        price = get_current_data()[stock].last_price
                        amount = min(available_cash, max_buy_amount) // (price * 100) * 100
                        if amount > 0:
                            log.info('买入 %s，数量：%d' % (stock, amount))
                            order(stock, amount)
                            available_cash -= amount * price
                            max_buy_amount -= amount * price