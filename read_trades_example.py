import pandas as pd

# 1) 从 CSV 读取（若同花顺导出为 Excel，请先在 Excel 另存为 CSV）
path = r"C:\Users\86137\同花顺\数据\Table.csv"  # <-- 替换为你的文件路径
try:
    # 同花顺导出在 Windows 常见编码为 'gbk'
    df = pd.read_csv(path, encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='utf-8', errors='ignore')
except Exception as e:
    print("读取 CSV 出错:", e)
    raise

print("列名:", df.columns.tolist())
print(df.head())

# 清洗示例：把可能的千分位逗号去掉并转为数值
for col in ['price', 'qty', '数量', '成交价']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float, errors='ignore')

# 2) 从剪贴板读取（如果你在同花顺中复制了表格）
try:
    df_clip = pd.read_clipboard()
    print("剪贴板读取成功，行数:", len(df_clip))
    print(df_clip.head())
except Exception as e:
    print("剪贴板读取失败:", e)

# 将 DataFrame 转换为标准 snapshot dict（示例）
def df_to_snapshot(df):
    # 需要根据你的 CSV 列名映射以下字段
    # 假设列名: 证券代码(code), 证券名称(name), 方向(side), 数量(qty), 成交价(price), 成交时间(time)
    mapping = {
        '代码': 'code', '证券代码': 'code',
        '名称': 'name', '证券名称': 'name',
        '方向': 'side', '买卖方向': 'side',
        '数量': 'qty', '成交数量': 'qty',
        '成交价': 'price', '价格': 'price',
        '成交时间': 'time', '时间': 'time'
    }
    df2 = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    # 做最小字段存在校验
    expected = ['code', 'name', 'side', 'qty', 'price']
    for e in expected:
        if e not in df2.columns:
            print(f"警告：缺少字段 {e}，请检查 CSV 列名或反馈样例给我")
    return df2

snapshot = df_to_snapshot(df)
print("snapshot preview:", snapshot.head().to_dict(orient='records')[:5])