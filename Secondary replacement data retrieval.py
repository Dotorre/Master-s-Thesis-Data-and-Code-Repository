import yfinance as yf
import pandas as pd

# 1. 设置股票列表和时间范围
tickers = ["WBD", "PARA", "APA", "WBA", "PCG", "VTRS", "BIIB"]
#WBD、PARA、APA、WBA、PCG、VTRS、DOW
start_date = "2014-01-01"
end_date = "2024-01-01"

# 2. 下载周频数据，自动复权
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1wk",
    group_by='ticker',
    auto_adjust=True,
    progress=False
)

# 3. 提取收盘价（构建一个通用的“收益率分析表格”）
close_df = pd.DataFrame()
for ticker in tickers:
    close_df[ticker] = data[ticker]['Close']

# 4. 保存为CSV文件
close_df.to_csv("C:/Users/Arine/Desktop/论文相关/task/过程文件/weekly_close_prices2.csv")
print("✅ Weekly close prices saved to 'weekly_close_prices3.csv'")


