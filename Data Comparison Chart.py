import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) 参数 & 数据
# =========================
CSV_PATH = "weekly_close_prices.csv"   
INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.001               # 0.1%
LOOKBACK = 52                          # 动量/反动量回顾窗（周）
USE_LOG_SCALE = True                   # 净值图 y 轴用对数刻度
TOPK = None                            

df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
df = df.sort_index()
tickers = df.columns.tolist()
dates = df.index
m = len(tickers)

# =========================
# 1) 工具函数
# =========================
def calculate_max_drawdown(value_series: pd.Series) -> float:
    peak = value_series.cummax()
    drawdown = (peak - value_series) / peak
    return float(drawdown.max())

def backtest_with_costs(weights_df: pd.DataFrame,
                        price_df: pd.DataFrame,
                        tc: float = TRANSACTION_COST,
                        initial_capital: float = INITIAL_CAPITAL):
    """
    通用回测器
    口径：cost_i^t = max(tc * |Δ_i^t|, 1) × 1{|Δ_i^t|>0}
    对齐：以“当期价格变动后的组合价值”V_pre为基准做再平衡
    """
    idx, cols = price_df.index, price_df.columns
    weights_df = weights_df.reindex(index=idx, columns=cols).fillna(0.0)

    V = pd.Series(index=idx, dtype=float)
    C = pd.Series(0.0, index=idx)
    V.iloc[0] = initial_capital

    # 用等权作为建仓种子，避免V_pre=0连锁
    w0 = weights_df.iloc[0]
    if w0.sum() == 0:
        w0 = pd.Series(np.ones(len(cols)) / len(cols), index=cols)
    w_prev = w0.copy()

    for i in range(1, len(idx)):
        t, tm1 = idx[i], idx[i-1]
        p_t, p_tm1 = price_df.loc[t], price_df.loc[tm1]

        # 上期持仓滚动到本期（再平衡前）
        rolled_value = w_prev * V.iloc[i-1] * (p_t / p_tm1)
        V_pre = rolled_value.sum()

        # 第1期的极端情形：若仍为0，视作从现金建仓
        if (i == 1) and (V_pre == 0):
            V_pre = V.iloc[i-1]
            rolled_value = pd.Series(0.0, index=cols)

        # 本期目标权重；行和=0则不交易（延用w_prev）
        w_tgt = weights_df.loc[t].fillna(0.0)
        if w_tgt.sum() == 0:
            w_tgt = w_prev.copy()

        target_value = w_tgt * V_pre
        trades = target_value - rolled_value

        cost_t = float((np.maximum(tc * trades.abs(), 1.0) * (trades.abs() > 0)).sum())
        C.iloc[i] = cost_t
        V.iloc[i] = V_pre - cost_t

        w_prev = w_tgt.copy()

    return V, C

def compute_bah_weights(price_df, init_weights=None, initial_capital: float = INITIAL_CAPITAL):
    """买入持有：期初建仓后不再交易，权重随价格自然漂移"""
    m = price_df.shape[1]
    if init_weights is None:
        init_weights = np.ones(m) / m
    init_prices = price_df.iloc[0].values
    shares = (initial_capital * init_weights) / init_prices
    val_mat = price_df.values * shares
    total_val = val_mat.sum(axis=1, keepdims=True)
    w = val_mat / total_val
    return pd.DataFrame(w, index=price_df.index, columns=price_df.columns)

def compute_crp_pre_weights(price_df):
    """CRP 在每期再平衡之前的“漂移权重” """
    idx = price_df.index
    m = price_df.shape[1]
    w_post_prev = np.ones(m) / m
    W_pre = []
    for i in range(1, len(idx)):
        p_t, p_tm1 = price_df.iloc[i].values, price_df.iloc[i-1].values
        w_pre_t = w_post_prev * (p_t / p_tm1)
        w_pre_t = w_pre_t / w_pre_t.sum()
        W_pre.append(w_pre_t)
        w_post_prev = np.ones(m) / m  # 本期末拉回等权
    W_pre = [np.ones(m) / m] + W_pre
    return pd.DataFrame(W_pre, index=idx, columns=price_df.columns)

def topk_plus_others(W: pd.DataFrame, k: int | None):
    if (k is None) or (k >= W.shape[1]):
        return W.copy()
    last = W.iloc[-1].abs().sort_values(ascending=False)
    cols = last.index[:k].tolist()
    W_top = W[cols].copy()
    W_top["Others"] = (1.0 - W_top.sum(axis=1)).clip(lower=0)
    return W_top

def plot_price_vs_weights(price_df: pd.DataFrame,
                          W: pd.DataFrame,
                          title_bottom: str,
                          outfile: str | None = None,
                          smooth_weeks: int = 0,
                          topk: int | None = TOPK):
   
    rel = price_df / price_df.iloc[0]
    Wp = W.rolling(smooth_weeks, min_periods=1).mean() if smooth_weeks > 0 else W
    Wp = topk_plus_others(Wp, topk)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for col in rel.columns:
        ax1.plot(rel.index, rel[col], linewidth=1.2, label=col)
    ax1.set_title("Stock Price / Initial Price")
    ax1.set_ylabel("Relative Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=4, fontsize=8, frameon=True)

    for col in Wp.columns:
        ax2.plot(Wp.index, Wp[col], linewidth=1.4, label=col)
    ax2.plot(Wp.index, np.zeros(len(Wp)), "k--", linewidth=1.0, label="Cash")
    ax2.set_title(title_bottom); ax2.set_ylabel("Weight"); ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", ncol=4, fontsize=8, frameon=True)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=220)
    plt.show()

# =========================
# 2) Final Combi（动量×反动量聚合）
# =========================
rolling_min = df.rolling(window=LOOKBACK, min_periods=1).min()
rolling_max = df.rolling(window=LOOKBACK, min_periods=1).max()

def compute_primitive_weight(price, p_min, p_max, kind='mom'):
    denom = (p_max - p_min).replace(0, np.nan)
    if kind == 'mom':
        w = (price - p_min) / denom
    else:
        w = (p_max - price) / denom
    return w.fillna(0.0)

primitive_weights = {}
for tkr in tickers:
    primitive_weights[(tkr, 'mom')]  = compute_primitive_weight(df[tkr], rolling_min[tkr], rolling_max[tkr], 'mom')
    primitive_weights[(tkr, 'anti')] = compute_primitive_weight(df[tkr], rolling_min[tkr], rolling_max[tkr], 'anti')

# 阶段1：单票内聚合
stage1_values, stage1_weights = {}, {}
for tkr in tickers:
    W_mom = pd.DataFrame(0.0, index=dates, columns=tickers);  W_mom[tkr] = primitive_weights[(tkr, 'mom')]
    W_ant = pd.DataFrame(0.0, index=dates, columns=tickers);  W_ant[tkr] = primitive_weights[(tkr, 'anti')]

    Vm, _ = backtest_with_costs(W_mom, df)
    Va, _ = backtest_with_costs(W_ant, df)
    stage1_values[tkr] = Vm  # 用于阶段2的价值加权

    wm = primitive_weights[(tkr, 'mom')]
    wa = primitive_weights[(tkr, 'anti')]
    den = (Vm + Va).replace(0, np.nan)
    theta = ((Vm * wm + Va * wa) / den).fillna(0.0)
    stage1_weights[tkr] = theta

# 阶段2：跨票聚合得到最终权重 W_final
V1_df = pd.DataFrame(stage1_values).reindex(index=dates, columns=tickers)
W_final = pd.DataFrame(0.0, index=dates, columns=tickers)
for t in dates:
    V1 = V1_df.loc[t]
    thetas = pd.Series({k: stage1_weights[k].loc[t] for k in tickers})
    num = (V1 * thetas)
    s = num.sum()
    W_final.loc[t] = (num / (s if s != 0 else 1.0)).values

# 对齐 & 行归一化，防止出现全零行
W_final = W_final.reindex(index=df.index, columns=df.columns)
W_final = W_final.div(W_final.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

# 最终：Combi 净值 & 成本
V_final, C_final = backtest_with_costs(W_final, df)

# =========================
# 3) 三个基准：BAH / Market / CRP
# =========================
# BAH：真·买入持有
weights_bah_true = compute_bah_weights(df)
relative_price = df / df.iloc[0]
V_bah = relative_price.dot(np.ones(m)/m) * INITIAL_CAPITAL
C_bah = pd.Series(0.0, index=dates)

# Market：每期价格占比（近似市值权重）
W_mkt = df.div(df.sum(axis=1), axis=0)
V_mkt, C_mkt = backtest_with_costs(W_mkt, df)

# CRP：每期等权对齐（权重图可视化建议用“再平衡前”版本）
W_crp_post = pd.DataFrame(1.0/m, index=dates, columns=tickers)
V_crp, C_crp = backtest_with_costs(W_crp_post, df)
W_crp_pre = compute_crp_pre_weights(df)  # 仅用于权重可视化（更“动”）

# =========================
# 4) 图表 —— 四策净值、累计成本、各策略两联图
# =========================
# 4.1 累计净值（对数刻度）
plt.figure(figsize=(11,6))
plt.plot(V_final, label='Final Combi Strategy', linewidth=2)
plt.plot(V_bah,   label='BAH (Buy-and-Hold)', linestyle='--')
plt.plot(V_mkt,   label='Market-Cap Weight',  linestyle=':')
plt.plot(V_crp,   label='CRP Equal-Weight (w/ cost)')
if USE_LOG_SCALE: plt.yscale("log")
plt.title('Fig. 4.1  Cumulative Net Value Comparison (weekly, costs included)', pad=10)
plt.xlabel('Date'); plt.ylabel('Portfolio Value (USD)' + (' [log scale]' if USE_LOG_SCALE else ''))
plt.legend(loc='upper left', ncol=2); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("fig_4_1_net_values.png", dpi=220); plt.show()

# 4.2 累计交易成本
C_final_cum, C_bah_cum, C_mkt_cum, C_crp_cum = C_final.cumsum(), C_bah.cumsum(), C_mkt.cumsum(), C_crp.cumsum()
plt.figure(figsize=(11,5.5))
plt.plot(C_final_cum, label='Final Combi')
plt.plot(C_bah_cum,   label='BAH (≈0)')
plt.plot(C_mkt_cum,   label='Market-Cap')
plt.plot(C_crp_cum,   label='CRP')
plt.title('Fig. 4.2  Cumulative Transaction Costs (fee per asset per period: max(tc*|Δ|, $1))', pad=10)
plt.xlabel('Date'); plt.ylabel('Cumulative Costs (USD)')
plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("fig_4_2_costs_cumulative.png", dpi=220); plt.show()

# 4.3 两联图：相对价格 + 权重折线（四张）
plot_price_vs_weights(df, weights_bah_true,
                      title_bottom="BAH Portfolio Weights Over Time (θ_it)",
                      outfile="fig_4_3a_bah_price_weights.png",
                      smooth_weeks=0, topk=TOPK)

plot_price_vs_weights(df, W_mkt,
                      title_bottom="Market-Cap Portfolio Weights Over Time (θ_it)",
                      outfile="fig_4_3b_market_price_weights.png",
                      smooth_weeks=0, topk=TOPK)

plot_price_vs_weights(df, W_final,
                      title_bottom="Final Combi Portfolio Weights Over Time (θ_it)",
                      outfile="fig_4_3c_combi_price_weights.png",
                      smooth_weeks=0, topk=TOPK)

# CRP：建议展示“再平衡前”的漂移权重；如需“再平衡后”等权，把 W_crp_pre 改成 W_crp_post
plot_price_vs_weights(df, W_crp_pre,
                      title_bottom="CRP (Pre-Rebalance) Weights Over Time (θ_it)",
                      outfile="fig_4_3d_crp_pre_price_weights.png",
                      smooth_weeks=0, topk=TOPK)

# =========================
# 5) 打印
# =========================
print("== Summary (Main Pool) ==")
print(f"  Final Combi:  total cost ${C_final_cum.iloc[-1]:.2f}, MDD {calculate_max_drawdown(V_final):.2%}")
print(f"  BAH:          total cost ${C_bah_cum.iloc[-1]:.2f}, MDD {calculate_max_drawdown(V_bah):.2%}")
print(f"  Market-Cap:   total cost ${C_mkt_cum.iloc[-1]:.2f}, MDD {calculate_max_drawdown(V_mkt):.2%}")
print(f"  CRP:          total cost ${C_crp_cum.iloc[-1]:.2f}, MDD {calculate_max_drawdown(V_crp):.2%}")
