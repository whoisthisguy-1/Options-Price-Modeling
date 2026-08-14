import matplotlib.pyplot as plt

def plotbody(title, filename, figsize, xlab="time", ylab="Dollars"):
    plt.figure(figsize=figsize)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.title(title)
    plt.show()
    plt.savefig(filename, dpi=150, bbox_inches="tight")

def plotlinegraph(title, filename,xlab, ylab,):
    plotbody(title, filename, (12, 15), xlab="time", ylab="Dollars")
    plt.plot(p_px["dt"], p_px["close"])


def plothistogram(title, filename, xlab, ylab ):
    plt.hist(p_px["ret"].dropna(), bins=35)



plt.figure(figsize=(12, 5))
plt.plot(p_px["dt"], p_px["close"])
plt.title("AXSM Closing Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.grid(True)
plt.savefig("charts/01_closing_price.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Histogram
plt.figure(figsize=(10, 5))
plt.hist(p_px["ret"].dropna(), bins=35)
plt.title("AXSM Daily Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("charts/02_returns_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Market vs Model for Calls
calls = p_opt[p_opt["type"] == "call"].sort_values("k")
plt.figure(figsize=(10, 5))
plt.plot(calls["k"], calls["mid"], marker="o", label="market mid")
plt.plot(calls["k"], calls["bs"], marker="o", label="BS value")
plt.title("AXSM Calls: Market vs Model")
plt.xlabel("Strike")
plt.ylabel("Option Value")
plt.legend()
plt.grid(True)
plt.savefig("charts/03_calls_market_vs_model.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Implied Volatility by strike price
plt.figure(figsize=(10, 5))

for typ in p_opt["type"].unique():
    tmp = p_opt[p_opt["type"] == typ]
    plt.scatter(tmp["k"], tmp["iv"], label=typ)
plt.axhline(ann_vol, linestyle="--", label="hist vol")
plt.title("Implied Volatility by Strike")
plt.xlabel("Strike")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
plt.savefig("charts/04_implied_vol_by_strike.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Risk Label Chart
risk_pd = p_opt.groupby(["type", "risk"]).size().reset_index(name="n")
labels = risk_pd["type"] + " - " + risk_pd["risk"]
plt.figure(figsize=(10, 5))
plt.bar(labels, risk_pd["n"])
plt.title("Risk Labels by Option Type")
plt.xlabel("Type and Risk")
plt.ylabel("Contracts")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("charts/05_risk_labels.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Break Even Distance Vs ITM
plt.figure(figsize=(10, 5))
plt.scatter(p_opt["be_dist"], p_opt["itm_prob"])
plt.title("Break-even Distance vs ITM Probability")
plt.xlabel("Break-even Distance")
plt.ylabel("ITM Probability")
plt.grid(True)
plt.savefig("charts/06_breakeven_vs_itm.png", dpi=150, bbox_inches="tight")
plt.show()

# %%


# %%
#Options Volume Chart
plt.figure(figsize=(12, 5))
plt.plot(p_vol["dt"], p_vol["tot_vol"], label="daily vol")
plt.plot(p_vol["dt"], p_vol["ma20"], label="20 day avg")
plt.scatter(
    p_vol[p_vol["weird_vol"] == 1]["dt"],
    p_vol[p_vol["weird_vol"] == 1]["tot_vol"],
    label="weird vol"
)



plt.title("AXSM Options Volume")
plt.xlabel("Date")
plt.ylabel("Contracts")
plt.legend()
plt.grid(True)
plt.savefig("charts/07_options_volume.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Throughput check
a = time.time()
n = po.count()
b = time.time()
sec = b - a
tp = n / sec if sec > 0 else np.nan


print("rows:", n)
print("seconds:", sec)
print("rows/sec:", tp)

# %%
#Scaling test
mults = [1, 10, 50, 100]
runs = []
for m in mults:
    big = po.crossJoin(spark.range(m).withColumnRenamed("id", "dup"))
    a = time.time()
    n = big.count()
    b = time.time()
    sec = b - a
    rows_sec = n / sec if sec > 0 else np.nan
    runs.append((m, n, sec, rows_sec))

# %%
#Benchmark Table
bench = pd.DataFrame(
    runs,
    columns=["mult", "rows", "secs", "rows_per_sec"]
)
bench

# %%
#Runtime Regression
x = bench["rows"].values.astype(float)
y = bench["secs"].values.astype(float)

# %%
#Runtime R^2
coef = np.polyfit(x, y, 1)
pred = coef[0] * x + coef[1]
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


print("runtime slope:", coef[0])
print("runtime R2:", r2)

# %%
#Runtime Chart
plt.figure(figsize=(8, 5))
plt.scatter(bench["rows"], bench["secs"])
plt.plot(bench["rows"], pred)
plt.title("Runtime Scaling")
plt.xlabel("Rows")
plt.ylabel("Seconds")
plt.grid(True)
plt.savefig("charts/08_runtime_scaling.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
#Saving outputs

