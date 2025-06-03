import efinance as ef
import pymysql
from datetime import datetime
import time

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'stock',
    'charset': 'utf8mb4'
}


def create_db_connection():
    """创建数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def get_realtime_data(stock_codes):
    """获取秒级交易数据"""
    data = ef.stock.get_quote_history(
        stock_codes,
        klt=1,  # 1表示1分钟级数据（最高精度）
        max_count=10  # 每次获取10条最新记录
    )
    # print(data)
    return data


def save_to_mysql(data):
    """保存数据到MySQL"""
    conn = create_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """INSERT INTO realtime_stock (
                stock_code, trade_time, price, price_change, 
                high, low, volume, amount, trade_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                price=VALUES(price), volume=VALUES(volume)"""

            for stock_code, df in data.items():
                for _, row in df.iterrows():
                    # 数据格式化
                    trade_time = datetime.strptime(
                        f"{row['日期']}",
                        '%Y-%m-%d %H:%M'
                    )
                    trade_type = 'buy' if row['涨跌幅'] > 0 else 'sell'

                    # 执行插入
                    cursor.execute(sql, (
                        stock_code,
                        trade_time,
                        row['收盘'],
                        row['涨跌幅'],
                        row['最高'],
                        row['最低'],
                        row['成交量'],
                        row['成交额'],
                        trade_type
                    ))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    STOCK_CODES = ['600519', '000001']  # 贵州茅台, 平安银行

    # while True:
    try:
        # 获取实时数据
        realtime_data = get_realtime_data(STOCK_CODES)

        # 存储到数据库
        print(realtime_data)
        if  realtime_data:
            save_to_mysql(realtime_data)
            print(f"{datetime.now()} 数据存储成功")

        # 每5秒采集一次
        time.sleep(5)

    except Exception as e:
        print(f"错误发生: {str(e)}")
        time.sleep(60)  # 出错后等待1分钟重试