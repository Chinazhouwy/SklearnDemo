import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# 获取股票数据
def get_stock_data(stock_code, start_date, end_date):
    """获取指定股票的历史数据"""
    stock_data = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
    print(stock_data.columns)
    stock_data.columns = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额',
       '换手率']
    stock_data['日期'] = pd.to_datetime(stock_data['日期'])
    stock_data.set_index('日期', inplace=True)
    return stock_data


# 计算弹性指标
def calculate_elasticity(stock_data, window=20):
    """计算股票的弹性指标 - 日内波动超过2%的频率"""
    stock_data['日内波动'] = (stock_data['最高'] - stock_data['最低']) / stock_data['开盘'] * 100
    stock_data['大幅波动'] = (stock_data['日内波动'] > 2).astype(int)
    stock_data['弹性指标'] = stock_data['大幅波动'].rolling(window=window).mean() * 100  # 转换为百分比
    return stock_data


# 计算技术指标 - 移动平均线
def calculate_ma(stock_data, short_window=5, long_window=20):
    """计算短期和长期移动平均线"""
    stock_data['MA_short'] = stock_data['收盘'].rolling(window=short_window).mean()
    stock_data['MA_long'] = stock_data['收盘'].rolling(window=long_window).mean()
    stock_data['金叉'] = (stock_data['MA_short'] > stock_data['MA_long']) & (
                stock_data['MA_short'].shift(1) <= stock_data['MA_long'].shift(1))
    stock_data['死叉'] = (stock_data['MA_short'] < stock_data['MA_long']) & (
                stock_data['MA_short'].shift(1) >= stock_data['MA_long'].shift(1))
    return stock_data


# 模拟交易
def simulate_trading(stock_data, initial_capital=50000, min_profit_per_share=200, min_shares=100):
    """模拟交易过程"""
    # 初始化参数
    capital = initial_capital
    shares = 0
    trades = []
    holding = False

    # 假设股票价格为10元，计算可购买的股数
    initial_price = stock_data.iloc[0]['收盘']
    max_shares = int(capital / initial_price / 100) * 100  # 确保是100的整数倍

    for i in range(len(stock_data)):
        current_date = stock_data.index[i]
        current_price = stock_data.iloc[i]['收盘']
        current_ma_short = stock_data.iloc[i]['MA_short']
        current_ma_long = stock_data.iloc[i]['MA_long']
        current_elasticity = stock_data.iloc[i]['弹性指标']

        # 跳过NaN值
        if pd.isna(current_ma_short) or pd.isna(current_ma_long) or pd.isna(current_elasticity):
            continue

        # 买入条件：弹性指标高且出现金叉
        if not holding and current_elasticity > 30 and stock_data.iloc[i]['金叉']:
            shares_to_buy = max_shares
            cost = shares_to_buy * current_price
            if cost <= capital:
                capital -= cost
                shares += shares_to_buy
                holding = True
                trades.append({
                    '日期': current_date,
                    '操作': '买入',
                    '价格': current_price,
                    '数量': shares_to_buy,
                    '花费': cost,
                    '剩余资金': capital,
                    '持有股数': shares
                })

        # 卖出条件：每手盈利超过200元或出现死叉
        elif holding:
            profit_per_share = current_price - initial_price
            profit_per_hand = profit_per_share * 100  # 一手100股

            # 部分卖出：每手盈利超过200元
            if profit_per_hand > min_profit_per_share and shares > min_shares:
                shares_to_sell = min(100, shares - min_shares)  # 至少保留一手
                revenue = shares_to_sell * current_price
                capital += revenue
                shares -= shares_to_sell
                trades.append({
                    '日期': current_date,
                    '操作': '卖出',
                    '价格': current_price,
                    '数量': shares_to_sell,
                    '收入': revenue,
                    '剩余资金': capital,
                    '持有股数': shares
                })

            # 全部卖出：出现死叉
            elif stock_data.iloc[i]['死叉'] and shares > 0:
                revenue = shares * current_price
                capital += revenue
                trades.append({
                    '日期': current_date,
                    '操作': '卖出',
                    '价格': current_price,
                    '数量': shares,
                    '收入': revenue,
                    '剩余资金': capital,
                    '持有股数': 0
                })
                shares = 0
                holding = False

    # 计算最终资产
    final_assets = capital
    if shares > 0:
        final_assets += shares * stock_data.iloc[-1]['收盘']

    # 计算收益率
    profit = final_assets - initial_capital
    profit_percentage = profit / initial_capital * 100

    return {
        'trades': pd.DataFrame(trades),
        'final_assets': final_assets,
        'profit': profit,
        'profit_percentage': profit_percentage
    }


# 主函数
def main():
    # 设置参数
    stock_code = "000001"  # 平安银行
    start_date = "20230101"
    end_date = datetime.now().strftime("%Y%m%d")
    initial_capital = 50000
    min_profit_per_share = 200  # 每手盈利200元

    # 获取数据
    print(f"获取 {stock_code} 的历史数据...")
    stock_data = get_stock_data(stock_code, start_date, end_date)

    # 计算指标
    print("计算弹性指标和移动平均线...")
    stock_data = calculate_elasticity(stock_data)
    stock_data = calculate_ma(stock_data)

    # 模拟交易
    print("模拟交易过程...")
    result = simulate_trading(stock_data, initial_capital, min_profit_per_share)

    # 输出结果
    print("\n交易记录:")
    if not result['trades'].empty:
        print(result['trades'])

    print(f"\n初始资金: {initial_capital}元")
    print(f"最终资产: {result['final_assets']:.2f}元")
    print(f"总盈利: {result['profit']:.2f}元")
    print(f"收益率: {result['profit_percentage']:.2f}%")

    # 可视化结果
    plt.figure(figsize=(14, 10))

    # 绘制股价和移动平均线
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['收盘'], label='收盘价')
    plt.plot(stock_data.index, stock_data['MA_short'], label='5日均线')
    plt.plot(stock_data.index, stock_data['MA_long'], label='20日均线')

    # 标记买卖点
    if not result['trades'].empty:
        buy_signals = result['trades'][result['trades']['操作'] == '买入']
        sell_signals = result['trades'][result['trades']['操作'] == '卖出']
        plt.scatter(buy_signals['日期'], [stock_data.loc[date, '收盘'] for date in buy_signals['日期']],
                    marker='^', color='g', s=100, label='买入')
        plt.scatter(sell_signals['日期'], [stock_data.loc[date, '收盘'] for date in sell_signals['日期']],
                    marker='v', color='r', s=100, label='卖出')

    plt.title(f'{stock_code} 股价和交易信号')
    plt.legend()

    # 绘制弹性指标
    plt.subplot(3, 1, 2)
    plt.plot(stock_data.index, stock_data['弹性指标'], color='purple')
    plt.axhline(y=30, color='r', linestyle='--')
    plt.title('弹性指标 (日内波动>2%的频率)')

    # 绘制交易资产变化
    if not result['trades'].empty:
        plt.subplot(3, 1, 3)
        trade_dates = result['trades']['日期'].tolist()
        trade_values = [initial_capital]

        for i in range(len(result['trades'])):
            if result['trades'].iloc[i]['操作'] == '买入':
                trade_values.append(trade_values[-1] - result['trades'].iloc[i]['花费'])
            else:
                trade_values.append(trade_values[-1] + result['trades'].iloc[i]['收入'])

        plt.plot(trade_dates, trade_values, color='blue')
        plt.title('交易资产变化')

    plt.tight_layout()
    plt.savefig(f'{stock_code}_trading_result.png')
    plt.show()


if __name__ == "__main__":
    main()
