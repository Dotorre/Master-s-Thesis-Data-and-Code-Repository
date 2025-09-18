import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%% 参数约定
# -----------------------------
CSV_PATH = 'weekly_close_prices2.csv'   
INITIAL_CAPITAL = 10_000              # 初始资金假设
TRANSACTION_COST = 0.001              # 成本
LOOKBACK = 52                         # 回顾期week：英美年交易日共252（除去休息日），一年有52周


#%% 读取数据
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
tickers = df.columns.tolist()
m = len(tickers)

#%% ===动量/反动量策略应用===


#预计算滚动最值（判断动量/反动量位置）
rolling_min = df.rolling(window=LOOKBACK, min_periods=1).min()
rolling_max = df.rolling(window=LOOKBACK, min_periods=1).max()

#%%计算权重
def compute_primitive_weight(price, p_min, p_max, kind='mom'):
    """
    price, p_min, p_max: pd.Series，同索引
    kind: 'mom' or 'anti'
    返回一个 Series
    """
    denom = (p_max - p_min).replace(0, np.nan)
    if kind == 'mom':
        w = (price - p_min) / denom
    else:
        w = (p_max - price) / denom
    return w.fillna(0)               # .fillna(0)将缺失值指定为0

#%%计算净值
def backtest_strategy(weights_df, price_df):
    """
    weights_df: 每列一个 ticker，每行是时点 t 的目标配置权重（sum ≤ 1）
    price_df: 相同形状的收盘价 DataFrame
    返回：每周滚动计算的组合净值
    """
    dates = price_df.index
    V = pd.Series(index=dates, dtype=float)
    V.iloc[0] = INITIAL_CAPITAL
    prev_w = pd.Series(0.0, index=tickers)
    
    for i in range(1, len(dates)):
        today, yesterday = dates[i], dates[i-1]
        p_today = price_df.loc[today]
        p_yest  = price_df.loc[yesterday]
        
        # 目标权重算出买入/卖出额
        target_value = weights_df.loc[today] * V.loc[yesterday]
        # 上周持仓滚动到今天的市值
        rolled_value = prev_w * V.loc[yesterday] * (p_today / p_yest)
        trades = target_value - rolled_value
        
        # 交易成本
        cost = (trades.abs().clip(lower=1) * TRANSACTION_COST).sum()
        
        # 当周 PnL
        pnl = (prev_w * V.loc[yesterday] * (p_today/p_yest - 1)).sum()
        V.iloc[i] = V.iloc[i-1] + pnl - cost
        
        prev_w = weights_df.loc[today].copy()
    return V

#%% 调用定义的函数进行实际计算权重&净值
# -----------------------------
primitive_weights = {}
primitive_values  = {}

for ticker in tickers:
    # 计算单支股票在每周的原始 mom/anti 权重
    w_mom  = compute_primitive_weight(df[ticker], rolling_min[ticker], rolling_max[ticker], 'mom')
    w_anti = compute_primitive_weight(df[ticker], rolling_min[ticker], rolling_max[ticker], 'anti')
    primitive_weights[(ticker,'mom')]  = w_mom
    primitive_weights[(ticker,'anti')] = w_anti
    
    # 整合成 m 维权重矩阵
    W_mom  = pd.DataFrame(0.0, index=df.index, columns=tickers)
    W_anti = W_mom.copy()
    W_mom[ticker]  = w_mom
    W_anti[ticker] = w_anti
    
    # 回测(value)
    primitive_values[(ticker,'mom')]  = backtest_strategy(W_mom, df)
    primitive_values[(ticker,'anti')] = backtest_strategy(W_anti, df)

#%% 阶段 1 聚合（每支股票内合并）
stage1_weights = {}
stage1_values  = {}

for ticker in tickers:
    Vm = primitive_values[(ticker,'mom')]
    Va = primitive_values[(ticker,'anti')]
    wm = primitive_weights[(ticker,'mom')]
    wa = primitive_weights[(ticker,'anti')]
    
    # 按价值加权合并权重
    theta = (Vm * wm + Va * wa) / (Vm + Va)
    stage1_weights[ticker] = theta
    
    # 回测合并后策略
    W_comb = pd.DataFrame(0.0, index=df.index, columns=tickers)
    W_comb[ticker] = theta
    stage1_values[ticker] = backtest_strategy(W_comb, df)     #继续调用backtest_strategy算净值

#%% 阶段 2 聚合（跨股票合并）
dates = df.index
V1_df = pd.DataFrame(stage1_values)

# 最终权重表
W_final = pd.DataFrame(0.0, index=dates, columns=tickers)

for t in dates:
    V1 = V1_df.loc[t]                             # 阶段1净值
    thetas = pd.Series({i: stage1_weights[i].loc[t] for i in tickers})
    # 加权合并
    W_final.loc[t] = (V1 * thetas).values / V1.sum()

# 最终策略回测(value)
V_final = backtest_strategy(W_final, df)


#%% BAH & 市值组合

#BAH 等权买入持有（无交易成本）
W_bah = pd.DataFrame(1/m, index=dates, columns=tickers)
V_bah = backtest_strategy(W_bah, df)

#市值组合（市场中七支股票自然指数）
mc_w = df.div(df.sum(axis=1), axis=0)
V_mkt = backtest_strategy(mc_w, df)

#%% CRP 等权持续再平衡
def backtest_crp_equal_weight(price_df,
                              initial_capital=INITIAL_CAPITAL,
                              tc=TRANSACTION_COST):
    """
    等权持续再平衡（CRP），每期目标权重为 1/m，按老师文件的成本定义：
    Cost_t = sum_i max(tc*|Δ_i^t|, 1) * 1{|Δ_i^t|>0}
    返回：净值序列（已扣除成本）
    """
    dates = price_df.index
    m = price_df.shape[1]
    theta_target = np.ones(m) / m  # 每期目标等权

    V_prev = initial_capital
    theta_prev = theta_target.copy()   # 上期配置（假定期初已按等权建仓，不计初始费用）
    V_list = [initial_capital]

    for t in range(1, len(dates)):
        p_t   = price_df.iloc[t].values
        p_tm1 = price_df.iloc[t-1].values

        # 本期收益（未扣成本）
        Rt = (p_t - p_tm1) / p_tm1
        V_now = V_prev * (1.0 + np.dot(theta_target, Rt))

        # 计算调仓金额 Δ_i^t 并计成本（严格按照“max(tc*|Δ|, 1)”）
        delta  = theta_target * V_now - theta_prev * V_prev * (p_t / p_tm1)
        cost_t = np.sum(np.maximum(tc * np.abs(delta), 1.0) * (np.abs(delta) > 0))

        # 扣除成本后的净值，更新持仓为目标等权
        V_now_net = V_now - cost_t
        V_list.append(V_now_net)
        V_prev = V_now_net
        theta_prev = theta_target.copy()

    return pd.Series(V_list, index=dates)

# 运行 CRP 回测
V_crp = backtest_crp_equal_weight(df)

#%%9. 绘图对比
plt.figure(figsize=(10,6))
plt.plot(V_final, label='Final Combi Strategy', linewidth=2)
plt.plot(V_bah,   label='BAH Equal-Weight',   linestyle='--')
plt.plot(V_mkt,   label='Market-Cap Weight',  linestyle=':')
plt.plot(V_crp,   label='CRP Equal-Weight (w/ cost)')

plt.title('Cumulative Net Value Comparison', fontsize=14, pad=10)
plt.xlabel('Date'); plt.ylabel('Log Portfolio Value (USD)')
plt.legend(); plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # ← 上方预留 5% 给标题
plt.show()


#%% 最大回撤计算函数
def calculate_max_drawdown(value_series):
    peak = value_series.cummax()
    drawdown = (peak - value_series) / peak
    return drawdown.max()

#%% 三种（现改为四种）策略的最大回撤
mdd_final = calculate_max_drawdown(V_final)
mdd_bah   = calculate_max_drawdown(V_bah)
mdd_mkt   = calculate_max_drawdown(V_mkt)
mdd_crp   = calculate_max_drawdown(V_crp) 

#%% 打印结果
print("最大回撤（Maximum Drawdown）结果：")
print(f"Final Combi Strategy: {mdd_final:.2%}")
print(f"BAH Equal-Weight:     {mdd_bah:.2%}")
print(f"Market-Cap Weight:    {mdd_mkt:.2%}")
print(f"CRP Equal-Weight:     {mdd_crp:.2%}")  

