import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "stocks",
    "user": "postgres",
    "password": "password"
}

# 股票列表（A股需加.SS后缀）
STOCKS = ["600519.SS", "000001.SZ", "601318.SS", "000333.SZ"]


def fetch_second_data(symbol, start_dt, end_dt):
    """获取秒级交易数据"""
    try:
        ticker = yf.Ticker(symbol)
        # 获取1分钟数据（yfinance最小粒度）
        df = ticker.history(
            start=start_dt,
            end=end_dt,
            interval="1m",
            prepost=True  # 包含盘前盘后
        )

        # 转换为秒级处理（模拟秒级）
        df = df.resample("s").ffill()

        # 添加股票代码列
        df["symbol"] = symbol
        return df.reset_index()

    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None


def save_to_db(data, symbol):
    """批量存储到TimescaleDB"""
    if data is None or data.empty:
        return

    print(data.columns)
    # print(f"data:{data.head()}")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 准备数据格式
        records = [
            (
                row["symbol"],
                row["Datetime"].to_pydatetime(),
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"],
                0,
                row.get("Dividends", 0),
                row.get("Stock Splits", 1)
            )
            for _, row in data.iterrows()
        ]


        # 批量插入（每秒约5000条）
        execute_values(
            cursor,
            """INSERT INTO stock_ticks (
                symbol, timestamp, open, high, low, close, 
                volume, adj_close, dividends, stock_splits
            ) VALUES %s
            ON CONFLICT (symbol, timestamp) DO NOTHING""",
            records
        )

        conn.commit()
        print(f"✅ {symbol} 插入 {len(records)} 条数据")

    except Exception as e:
        print(f"数据库写入失败: {e}")
    finally:
        if conn: conn.close()


def worker(symbol):
    """多线程任务处理"""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)  # 获取最近7天数据

    print(f"开始处理 {symbol}...")
    data = fetch_second_data(symbol, start_dt, end_dt)
    if data is not None:
        save_to_db(data, symbol)


if __name__ == "__main__":
    start_time = time.time()

    # 多线程处理（建议线程数=CPU核心数×2）
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(worker, STOCKS)

    print(f"总耗时: {time.time() - start_time:.2f}秒")