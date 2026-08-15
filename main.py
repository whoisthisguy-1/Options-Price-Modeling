# %%

# %%
import math
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
import pyspark
import scipy.stats as stats
import yfinance
from pyspark.sql import SparkSession
# %%
from pyspark.sql.functions import (avg, col, count, datediff, expr, isnan, lag,
                                   lit, log)
from pyspark.sql.functions import max as smax
from pyspark.sql.functions import min as smin
from pyspark.sql.functions import stddev
from pyspark.sql.functions import sum as ssum
from pyspark.sql.functions import to_date, udf, when
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

import graphing.charts as charts

# %%


# %%
spark = SparkSession.builder.appName("BS_project").getOrCreate()
sc = spark.sparkContext
print("Spark version:", spark.version)

# %%
# Stock Price Data for a year timeframe
px = pd.read_excel("inputs/AXSM_1Year_pricing.xlsx")
px.head()

# %%
# Data Column Cleaning and conversion to Spark
px.columns = px.columns.astype(str).str.replace("\xa0", "", regex=False).str.strip()
print(px.columns.tolist())
pxdf = spark.createDataFrame(px)
pxdf = pxdf.select(
    to_date(col("Date")).alias("dt"),
    col("Open").cast("double").alias("open"),
    col("High").cast("double").alias("high"),
    col("Low").cast("double").alias("low"),
    col("Close").cast("double").alias("close"),
    col("Adj Close").cast("double").alias("adj"),
    col("Volume").cast("double").alias("stk_vol"),
)

# %%
# Checking data in Spark
pxdf.printSchema()
pxdf.show(5)
print("price rows:", pxdf.count())

# %%
# Date window
w = Window.orderBy("dt")

# %%
# Returns the previous close alsongside daily log returns
rets = pxdf.withColumn("prev", lag("close").over(w))
rets.show(5)
rets = rets.withColumn("ret", log(col("close") / col("prev"))).dropna(subset=["ret"])

rets.show(5)
print("return rows:", rets.count())

# %%
# Calculation for return summary
st = rets.agg(
    avg("ret").alias("avg_ret"),
    stddev("ret").alias("day_vol"),
    smin("dt").alias("start_dt"),
    smax("dt").alias("end_dt"),
    count("*").alias("n"),
).collect()[0]

# %%
# Values for Volatility
avg_ret = float(st["avg_ret"])
day_vol = float(st["day_vol"])
ann_vol = day_vol * math.sqrt(252)
print("daily vol:", day_vol)
print("annual vol:", ann_vol)

# %%
# Most recent stock price
last = pxdf.orderBy(col("dt").desc()).select("dt", "close").first()
val_dt = last["dt"]
spot = float(last["close"])
print("valuation date:", val_dt)
print("spot:", spot)

# %%
# Simple Sumamry of the stock
print("start:", st["start_dt"])
print("end:", st["end_dt"])
print("n:", st["n"])
print("avg daily return:", avg_ret)
print("annual vol:", ann_vol)

# %%
# establishing rolling volatility windows for 20 monthly trading days and 60 quaterly trading days
w20 = Window.orderBy("dt").rowsBetween(-19, 0)
w60 = Window.orderBy("dt").rowsBetween(-59, 0)
rv = rets.withColumn("vol20", stddev("ret").over(w20) * math.sqrt(252)).withColumn(
    "vol60", stddev("ret").over(w60) * math.sqrt(252)
)
rv.select("dt", "close", "ret", "vol20", "vol60").show(10)

# %%
# The Options matrix
raw = pd.read_excel("inputs/AXSM_Recent_Spot_Pricing.xlsx", header=None)
raw.head(10)

# %%
rows = []
sec = None
hdr = None

# %%
# Parsing Calls and Puts
for _, r in raw.iterrows():
    first = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
    if first == "Calls":
        sec = "call"
        continue
    if first == "Puts":
        sec = "put"
        continue
    if first == "Contract Name":
        hdr = [str(x).strip() for x in r.tolist()]
        continue
    if sec in ["call", "put"] and first.startswith("AXSM"):
        x = dict(zip(hdr, r.tolist()))
        x["type"] = sec
        rows.append(x)

# %%
# Options Pandas dataframe
opt = pd.DataFrame(rows)
print("contracts found:", len(opt))
opt.head()

# %%
# Numeric options columns + clean
num_cols = [
    "Strike",
    "Last Price",
    "Bid",
    "Ask",
    "Change",
    "% Change",
    "Volume",
    "Open Interest",
    "Implied Volatility",
]
for c in num_cols:
    if c in opt.columns:
        opt[c] = pd.to_numeric(
            opt[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("-", "", regex=False),
            errors="coerce",
        )
opt.head()

# %%
# Fixing implied Volaility formatting
if "Implied Volatility" in opt.columns:
    opt["Implied Volatility"] = np.where(
        opt["Implied Volatility"] > 3,
        opt["Implied Volatility"] / 100,
        opt["Implied Volatility"],
    )
opt[["Implied Volatility"]].head()


# %%
# Expiration date
def get_exp(cn):
    s = str(cn)
    m = re.search(r"AXSM(\d{6})", s)

    if not m:
        return np.nan
    return pd.to_datetime(m.group(1), format="%y%m%d", errors="coerce").date()


opt["exp"] = opt["Contract Name"].apply(get_exp)

opt[["Contract Name", "exp"]].head()

# %%
# Properly Naming the options columns
opt = opt.rename(
    columns={
        "Contract Name": "name",
        "Last Trade Date": "last_trade",
        "Last Trade Date (EDT)": "last_trade",
        "Strike": "k",
        "Last Price": "last_px",
        "Bid": "bid",
        "Ask": "ask",
        "Change": "chg",
        "% Change": "pct_chg",
        "Volume": "opt_vol",
        "Open Interest": "oi",
        "Implied Volatility": "iv",
    }
)
opt.head()

# %%
# converting the options to spark and cleanign the options dates
opt["exp"] = pd.to_datetime(opt["exp"])
if "last_trade" in opt.columns:
    opt["last_trade"] = pd.to_datetime(opt["last_trade"], errors="coerce")
opt.head()
optdf = spark.createDataFrame(opt)
optdf.printSchema()
optdf = spark.createDataFrame(opt)
optdf.printSchema()

# %%
# if avaible clean last trade date
if "last_trade" in optdf.columns:
    optdf = optdf.withColumn("last_trade", to_date(col("last_trade")))

# %%
# TTM
optdf = optdf.withColumn("t", datediff(col("exp"), lit(val_dt)) / 365)

# %%
# Optimizatoin for data layout
optdf = optdf.repartition("type", "exp")
optdf.show(10, truncate=False)
print("option rows:", optdf.count())

# %%
type_ct = optdf.groupBy("type").agg(count("*").alias("n")).orderBy("type")
type_ct.show()

# %%
# Average option statistics
type_sum = (
    optdf.groupBy("type")
    .agg(
        avg("k").alias("avg_strike"),
        avg("bid").alias("avg_bid"),
        avg("ask").alias("avg_ask"),
        avg("iv").alias("avg_iv"),
        avg("oi").alias("avg_oi"),
        count("*").alias("n"),
    )
    .orderBy("type")
)
type_sum.show()


# %%
# cdF function
def n_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# %%
# helper cell for d1 and d2 in BS
def d_vals(s, k, t, r, v):
    if None in [s, k, t, r, v]:
        return None, None
    s, k, t, r, v = map(float, [s, k, t, r, v])
    if s <= 0 or k <= 0 or t <= 0 or v <= 0:
        return None, None
    d1 = (math.log(s / k) + (r + 0.5 * v**2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return d1, d2


# %%
# Pricing function for Black-Scholes
def bs_px(s, k, t, r, v, typ):
    d1, d2 = d_vals(s, k, t, r, v)
    if d1 is None:
        return None
    s, k, t, r = map(float, [s, k, t, r])
    if typ == "call":
        return s * n_cdf(d1) - k * math.exp(-r * t) * n_cdf(d2)
    if typ == "put":
        return k * math.exp(-r * t) * n_cdf(-d2) - s * n_cdf(-d1)
    return None


# %%
# Delta Function
def bs_del(s, k, t, r, v, typ):
    d1, _ = d_vals(s, k, t, r, v)
    if d1 is None:
        return None
    if typ == "call":
        return n_cdf(d1)
    if typ == "put":
        return n_cdf(d1) - 1
    return None


# %%
# In the Money Probability
def itm_p(s, k, t, r, v, typ):
    _, d2 = d_vals(s, k, t, r, v)
    if d2 is None:
        return None
    if typ == "call":
        return n_cdf(d2)
    if typ == "put":
        return n_cdf(-d2)
    return None


# %%
bs_px_udf = udf(bs_px, DoubleType())
bs_del_udf = udf(bs_del, DoubleType())
itm_udf = udf(itm_p, DoubleType())

# %%
# Risk free rate
rf = 0.045

# %%
# Base Pricing columns with black-scholes value, delta, and in the money probability
po = (
    optdf.withColumn("s", lit(float(spot)))
    .withColumn("rf", lit(float(rf)))
    .withColumn("hist_vol", lit(float(ann_vol)))
    .withColumn("mid", (col("bid") + col("ask")) / 2)
)
po.select("type", "k", "bid", "ask", "mid").show(10)
po = po.withColumn(
    "bs",
    bs_px_udf(col("s"), col("k"), col("t"), col("rf"), col("hist_vol"), col("type")),
)
po.select("type", "k", "mid", "bs").show(10)
po = po.withColumn(
    "delta",
    bs_del_udf(col("s"), col("k"), col("t"), col("rf"), col("hist_vol"), col("type")),
)
po.select("type", "k", "delta").show(10)
po = po.withColumn(
    "itm_prob",
    itm_udf(col("s"), col("k"), col("t"), col("rf"), col("hist_vol"), col("type")),
)
po.select("type", "k", "itm_prob").show(10)

# %%
# Break even price and distance
po = po.withColumn(
    "be",
    when(col("type") == "call", col("k") + col("mid")).when(
        col("type") == "put", col("k") - col("mid")
    ),
)
po.select("type", "k", "mid", "be").show(10)


po = po.withColumn(
    "be_dist",
    when(col("type") == "call", (col("be") - col("s")) / col("s")).when(
        col("type") == "put", (col("s") - col("be")) / col("s")
    ),
)
po.select("type", "k", "be", "be_dist").show(10)

# %%
# Creating Scenario pricing
po = po.withColumn("s_up10", col("s") * 1.10)
po = po.withColumn("s_dn10", col("s") * 0.90)
po = po.withColumn("vol_up25", col("hist_vol") * 1.25)

# %%
# Pricing for being up 10% and down 10%
po = po.withColumn(
    "bs_up10",
    bs_px_udf(
        col("s_up10"), col("k"), col("t"), col("rf"), col("hist_vol"), col("type")
    ),
)
po = po.withColumn(
    "bs_dn10",
    bs_px_udf(
        col("s_dn10"), col("k"), col("t"), col("rf"), col("hist_vol"), col("type")
    ),
)

# %%
# Up Volatilty scenario
po = po.withColumn(
    "bs_vol_up",
    bs_px_udf(col("s"), col("k"), col("t"), col("rf"), col("vol_up25"), col("type")),
)

# %%
# Scenario returns
po = po.withColumn(
    "up10_ret", when(col("mid") > 0, (col("bs_up10") - col("mid")) / col("mid"))
).withColumn(
    "dn10_ret", when(col("mid") > 0, (col("bs_dn10") - col("mid")) / col("mid"))
)
po.select("type", "k", "mid", "bs_up10", "bs_dn10", "up10_ret", "dn10_ret").show(10)

# %%
# Bid ask spread, premium and Implied Volatility gap
po = (
    po.withColumn("spr", when(col("mid") > 0, (col("ask") - col("bid")) / col("mid")))
    .withColumn("prem", when(col("bs") > 0, (col("mid") - col("bs")) / col("bs")))
    .withColumn("iv_gap", col("iv") - col("hist_vol"))
)

po.select("type", "k", "spr", "prem", "iv_gap").show(10)

# %%
# Wide Spread Risk Flag
po = po.withColumn("f_spread", when(col("spr") > 0.30, 1).otherwise(0))

# %%
# Low open intrest risk flag
po = po.withColumn("f_oi", when(col("oi") < 25, 1).otherwise(0))

# %%
# low contract volume risk flag
po = po.withColumn("f_vol", when(col("opt_vol") < 10, 1).otherwise(0))

# %%
# Far distance to break even risk flag
po = po.withColumn("f_be", when(col("be_dist") > 0.15, 1).otherwise(0))

# %%
# Low probability of being in the money risk flag
po = po.withColumn("f_prob", when(col("itm_prob") < 0.25, 1).otherwise(0))

# %%
# Market Premium Risk flag
po = po.withColumn("f_prem", when(col("prem") > 0.50, 1).otherwise(0))

# %%
# High implied volatility risk flag
po = po.withColumn("f_iv", when(col("iv") > col("hist_vol") * 1.30, 1).otherwise(0))

# %%
# Final score calculation for all risk flags
po = po.withColumn(
    "score",
    col("f_spread")
    + col("f_oi")
    + col("f_vol")
    + col("f_be")
    + col("f_prob")
    + col("f_prem")
    + col("f_iv"),
)
po.select("type", "k", "score").show(10)

# %%
# Risk Label criteria
po = po.withColumn(
    "risk",
    when(col("score") >= 5, "avoid")
    .when(col("score") >= 3, "caution")
    .otherwise("review"),
)
po.select("type", "k", "score", "risk").show(20)

# %%
po.cache()

# %%
# top 30 riskiest contracts
po.select(
    "name",
    "type",
    "k",
    "mid",
    "bs",
    "iv",
    "spr",
    "oi",
    "opt_vol",
    "be_dist",
    "itm_prob",
    "score",
    "risk",
).orderBy(col("score").desc(), col("spr").desc()).show(30, truncate=False)

# %%
# groupBy risk summary
risk_sum = po.groupBy("type", "risk").agg(count("*").alias("n")).orderBy("type", "risk")
risk_sum.show()

# %%
# Average risk stats
avg_risk = po.groupBy("type").agg(
    avg("score").alias("avg_score"),
    avg("spr").alias("avg_spread"),
    avg("iv").alias("avg_iv"),
    avg("be_dist").alias("avg_be_dist"),
    avg("itm_prob").alias("avg_itm_prob"),
    count("*").alias("n"),
)
avg_risk.show()

# %%
# SQL views
po.createOrReplaceTempView("opts")
pxdf.createOrReplaceTempView("prices")
rets.createOrReplaceTempView("rets")
print("SQL views made")

# %%
# SQL risk summary
sql_risk = spark.sql("""
SELECT
    type,
    risk,
    COUNT(*) AS n,
    ROUND(AVG(score), 2) AS avg_score,
    ROUND(AVG(spr), 3) AS avg_spread,
    ROUND(AVG(itm_prob), 3) AS avg_itm_prob
FROM opts
GROUP BY type, risk
ORDER BY type, risk
""")
sql_risk.show()

# %%
# Top 10 riskiest SQL contracts
sql_top = spark.sql("""
SELECT
    name,
    type,
    k,
    mid,
    bs,
    iv,
    spr,
    oi,
    opt_vol,
    be_dist,
    itm_prob,
    score,
    risk
FROM opts
ORDER BY score DESC, spr DESC
LIMIT 10
""")
sql_top.show(truncate=False)

# %%
# price summary in SQL
sql_px = spark.sql("""
SELECT
    MIN(dt) AS start_dt,
    MAX(dt) AS end_dt,
    COUNT(*) AS n,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND(MIN(close), 2) AS min_close,
    ROUND(MAX(close), 2) AS max_close
FROM prices
""")
sql_px.show()

# %%
# Options volume csv data clean
vdf = spark.read.csv(
    "inputs/daily_volume_AXSM_2025-06-23_2026-06-23.csv", header=True, inferSchema=True
)
vdf.printSchema()
vdf.show(5)
vdf = vdf.select(
    expr("try_to_timestamp(`Trade Date`, 'M/d/yyyy')").cast("date").alias("dt"),
    col("Options Class").alias("cls"),
    col("Underlying").alias("ticker"),
    col("Product Type").alias("prod"),
    col("Exchange").alias("exch"),
    col("Volume").cast("double").alias("vol"),
)
vdf.show(5)

# %%
# Daily Volume
dvol = vdf.groupBy("dt", "ticker").agg(ssum("vol").alias("tot_vol")).orderBy("dt")
dvol.show(10)
print("daily volume rows:", dvol.count())

# %%
# Moving average window
vw = Window.partitionBy("ticker").orderBy("dt").rowsBetween(-19, 0)

# %%
# 20day average for volume
dvol = dvol.withColumn("ma20", avg("tot_vol").over(vw))
dvol.show(10)

# %%
# Flagging any abnormalities in volume by weird
dvol = dvol.withColumn(
    "weird_vol", when(col("tot_vol") > 2 * col("ma20"), 1).otherwise(0)
)
dvol.orderBy(col("tot_vol").desc()).show(10)

# %%
# Latest Volume
latest_vol = (
    dvol.orderBy(col("dt").desc())
    .limit(1)
    .select(col("dt").alias("vol_dt"), col("tot_vol"), col("ma20"), col("weird_vol"))
)
latest_vol.show()

# %%
# Joining volume to options
po2 = po.crossJoin(latest_vol)
po2.select(
    "name",
    "type",
    "k",
    "mid",
    "score",
    "risk",
    "vol_dt",
    "tot_vol",
    "ma20",
    "weird_vol",
).show(12, truncate=False)

# %%
# Pandas conversion of price
p_px = rv.select("dt", "close", "ret", "vol20", "vol60").toPandas()

# %%
# Pandas conversion of option
p_opt = po.select(
    "type",
    "k",
    "mid",
    "bs",
    "iv",
    "risk",
    "spr",
    "score",
    "be_dist",
    "itm_prob",
    "up10_ret",
    "dn10_ret",
).toPandas()

# %%/
# Pandas conversion of volume
p_vol = dvol.select("dt", "tot_vol", "ma20", "weird_vol").toPandas()

# %%
# Closing Price Chart
os.makedirs("charts", exist_ok=True)
plt.figure(figsize=(12, 5))
plt.plot(p_px["dt"], p_px["close"])
plt.title("AXSM Closing Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.grid(True)
plt.savefig("charts/01_closing_price.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Histogram
plt.figure(figsize=(10, 5))
plt.hist(p_px["ret"].dropna(), bins=35)
plt.title("AXSM Daily Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("charts/02_returns_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Market vs Model for Calls
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
# Implied Volatility by strike price
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
# Risk Label Chart
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
# Break Even Distance Vs ITM
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
# Options Volume Chart
plt.figure(figsize=(12, 5))
plt.plot(p_vol["dt"], p_vol["tot_vol"], label="daily vol")
plt.plot(p_vol["dt"], p_vol["ma20"], label="20 day avg")
plt.scatter(
    p_vol[p_vol["weird_vol"] == 1]["dt"],
    p_vol[p_vol["weird_vol"] == 1]["tot_vol"],
    label="weird vol",
)


plt.title("AXSM Options Volume")
plt.xlabel("Date")
plt.ylabel("Contracts")
plt.legend()
plt.grid(True)
plt.savefig("charts/07_options_volume.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Throughput check
a = time.time()
n = po.count()
b = time.time()
sec = b - a
tp = n / sec if sec > 0 else np.nan


print("rows:", n)
print("seconds:", sec)
print("rows/sec:", tp)

# %%
# Scaling test
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
# Benchmark Table
bench = pd.DataFrame(runs, columns=["mult", "rows", "secs", "rows_per_sec"])
bench

# %%
# Runtime Regression
x = bench["rows"].values.astype(float)
y = bench["secs"].values.astype(float)

# %%
# Runtime R^2
coef = np.polyfit(x, y, 1)
pred = coef[0] * x + coef[1]
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


print("runtime slope:", coef[0])
print("runtime R2:", r2)

# %%
# Runtime Chart
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
# Saving outputs
po.toPandas().to_csv("outputs/axsm_contract_risk_scores.csv", index=False)
print("saved axsm_contract_risk_scores.csv")
po.toPandas().to_csv("outputs/axsm_contract_risk_scores.csv", index=False)
print("saved axsm_contract_risk_scores.csv")
bench.to_csv("outputs/axsm_scaling_bench.csv", index=False)
print("saved axsm_scaling_bench.csv")
