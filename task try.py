import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("weekly_close_prices.csv", index_col=0, parse_dates=True)

#%% 设定初始资金
V0 = 10000
m = len(data.columns)
theta = np.ones(m) / m  # 均分每个资产

# 所有股票除以初始价格，得到相对收益 X(t)/X(0)
relative_price = data / data.iloc[0]

# Buy and Hold Portfolio Value: V_t = sum(theta_i * X_i(t)/X_i(0))
bah_values = relative_price.dot(theta) * V0

# 取对数
log_bah = np.log(bah_values)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(log_bah, label="Buy and Hold (log)")
plt.title("Buy and Hold Portfolio (Log Value)")
plt.xlabel("Date")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

#%% 初始化
V_rebalanced = [V0]
cost_total = 0
n_periods = len(data)
dates = data.index
returns = data.pct_change().dropna()

# 每期的恒定权重
theta_crp = np.ones(m) / m

# 初始化上期配置
theta_prev = theta_crp.copy()
V_prev = V0

# 计算组合价值序列和成本
for t in range(1, n_periods):
    Rt = (data.iloc[t] - data.iloc[t - 1]) / data.iloc[t - 1]  # 当期收益率
    V_now = V_prev * (1 + np.dot(theta_crp, Rt))  # 当前组合价值

    # 计算调仓的绝对金额变动 Δ_i^t
    delta = theta_crp * V_now - theta_prev * V_prev * (data.iloc[t] / data.iloc[t - 1])
    cost_t = np.sum(np.maximum(0.001 * np.abs(delta), 1) * (np.abs(delta) > 0))  # 本期成本
    cost_total += cost_t

    # 保存并更新
    V_rebalanced.append(V_now)
    V_prev = V_now
    theta_prev = theta_crp.copy()

# 转为 Series 并计算 log 值
log_crp = np.log(pd.Series(V_rebalanced, index=dates[:len(V_rebalanced)]))

plt.figure(figsize=(10, 5))
plt.plot(log_bah, label="Buy and Hold (log)")
plt.plot(log_crp, label="Constant Rebalanced (log, w/ cost)")
plt.title("Portfolio Comparison: BAH vs CRP")
plt.xlabel("Date")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

print(f"Total Transaction Cost (CRP): ${cost_total:.2f}")


#%% Market portfolio value at each time step
market_value = data.sum(axis=1) / data.iloc[0].sum()

# 把它也换成 log value，和前两者一致
log_market = np.log(market_value * V0)  # 保持初始价值一致

plt.figure(figsize=(10, 5))
plt.plot(log_bah, label="Buy and Hold (log)")
plt.plot(log_crp, label="Constant Rebalanced (log, w/ cost)")
plt.plot(log_market, label="Market Index (log)")
plt.title("Portfolio Log-Value Comparison")
plt.xlabel("Date")
plt.ylabel("Log Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()


# 1. 计算相对价格 X(t)/X(0)
relative_price = data / data.iloc[0]

# 2. 计算 BAH 每个时刻的权重 θ_it
weights_bah = relative_price.div(relative_price.sum(axis=1), axis=0)

cash = 1 - weights_bah.sum(axis=1)

# 4. 画图（上下布局）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 上图：股票相对价格
relative_price.plot(ax=ax1)
ax1.set_title("Stock Price / Initial Price")
ax1.set_ylabel("Relative Price")
ax1.grid(True)

# 下图：BAH 每只股票的权重随时间变化——theta
weights_bah.plot(ax=ax2)
cash.plot(ax=ax2, color='black', linestyle='--', label='Cash')
ax2.set_title("BAH Portfolio Weights Over Time (θ_it)")
ax2.set_ylabel("Weight")
ax2.set_xlabel("Date")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

#%% 算结果
def calculate_mdd(series):
    peak = series.cummax()
    drawdown = (peak - series) / peak
    return drawdown.max()

mdd_bah = calculate_mdd(bah_values)
mdd_crp = calculate_mdd(pd.Series(V_rebalanced, index=data.index[:len(V_rebalanced)]))
mdd_market = calculate_mdd(market_value * V0)

print(f"Maximum Drawdown (BAH):     {mdd_bah:.2%}")
print(f"Maximum Drawdown (CRP):     {mdd_crp:.2%}")
print(f"Maximum Drawdown (Market):  {mdd_market:.2%}")

