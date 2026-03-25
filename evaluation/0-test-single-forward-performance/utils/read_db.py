import pandas as pd
import sqlite3

CSV_SAVE_PATH = "./result/read_db.csv"

def read_sqlite_with_pandas(db_path):
    """使用 pandas 读取 SQLite 数据库"""
    # 直接读取整个表
    df = pd.read_sql_query("SELECT * FROM records", sqlite3.connect(db_path))
    
    # 查看数据
    print("数据内容:")
    print(df)
    
    # 保存到 CSV 文件
    df.to_csv(CSV_SAVE_PATH, index=False)
    print(f"数据已保存到 {CSV_SAVE_PATH}")
    
    print("\n数据统计:")
    print(df.describe())
    
    # 按模型分组分析
    print("\n按模型分组:")
    grouped = df.groupby('tag')
    for model, group in grouped:
        print(f"\n模型: {model}")
        print(f"测试次数: {len(group)}")
        print(f"平均预填充时间: {group['avg_prefill_time_usage'].mean():.2f} ms")
        print(f"平均解码时间: {group['avg_decoding_time_usage'].mean():.2f} ms")
    
    return df

# 使用示例
if __name__ == "__main__":
    df = read_sqlite_with_pandas("db-identical-req.sqlite")

