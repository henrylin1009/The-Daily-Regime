# Macro Intelligence Platform — 進度報告

**報告日期：** 2026-05-27  
**專案路徑：** `analysis/`  
**定位：** 每日宏觀情報 brief；一條指令產出 HTML，供長期指數投資人快速掌握「大環境有沒有變」。

---

## 1. 總體狀態

| 階段 | 狀態 | 說明 |
|------|------|------|
| **Phase 1（MVP）** | ✅ 完成 | 資料收集、紅綠燈、歷史 regime、Gemini 敘事、HTML brief |
| **Phase 2（量化擴充）** | ✅ 大致完成 | 六面板因子歸因、S&P 成分貢獻、regime 統計、通知、Web UI、CI |
| **日常可用性** | ✅ 已驗證 | `python run.py` 約 30–40 秒；最新 brief：`output/brief_YYYY-MM-DD.html` |

**測試：** `pytest tests/` → 9 passed  
**快取：** `data/raw/` 約 31 個 CSV（含 SPY 成分價、持股、FF 因子等）

---

## 2. 流水線（`run.py`）

```
Stage 1   collect_all          → data/raw/*.csv
Stage 2   compute_signals      → 紅綠燈 + 指標表
Stage 2b  factor_attrib        → Panel A–D（迴歸歸因）
Stage 2c  index_contrib        → Panel E–F（持股拆解）併入 factor_attrib
Stage 3   history_match        → 1994+ 相似 regime
Stage 3b  regime_stats         → 持續性 / 回撤統計
Stage 4   Gemini narrative     → 宏觀摘要（可 --skip-llm）
Render    daily_brief.html      → output/brief_YYYY-MM-DD.html
Optional  notify               → Email / Telegram（--notify）
```

### CLI 選項

| 旗標 | 用途 |
|------|------|
| `--force-refresh` | 強制重抓所有資料 |
| `--skip-llm` | 略過 Gemini，使用 placeholder 敘事 |
| `--date YYYY-MM-DD` | 指定輸出檔日期 |
| `--notify` | 發送 Email / Telegram |
| `--skip-notify` | 即使設了 `NOTIFY_CHANNELS` 也不發送 |

---

## 3. 每日 Brief 內容

| 區塊 | 模組 | 內容 |
|------|------|------|
| Traffic Lights | `src/indicators.py` | 衰退 / 通膨 / 金融壓力 三燈 |
| Macro Summary | `src/analyst.py` | Gemini 敘事（或離線 placeholder） |
| SPY Return Attribution | `src/factor_attrib.py`, `src/index_contrib.py` | 六個歸因面板（見下） |
| Regime Persistence | `src/regime_stats.py` | 相似 regime 後續表現統計 |
| Closest Historical Match | `src/history_match.py` | 最像時期 + 其他相似期 + `what_happened` |
| Indicator Details | `src/indicators.py` | 各序列數值、分位、趨勢、As of 日期 |

### SPY 歸因六面板

| 面板 | 方法 | 因子 / 標的 |
|------|------|-------------|
| **A** | 63 日 beta × 21 日因子變動 | GLD、2y10y、信用利差、油、DXY、VIX |
| **B** | 月頻 CRR 風格 | 工業產出、CPI、期限、違約、失業 |
| **C** | Fama–French 5 + Momentum | Mkt-RF, SMB, HML, RMW, CMA, Mom |
| **D** | 板塊 ETF | XLK, XLF, XLV, XLE, XLU, XLI |
| **E** | 持股權重 × 個股報酬（1 日） | Top 10 貢獻者，依 **Contribution** 排序 |
| **F** | 同上（約 21 交易日） | 同上 |

**Panel E/F 欄位說明**

| 欄位 | 意義 |
|------|------|
| Stock return | 個股在區間內漲跌幅（%） |
| Weight | SPY 持股權重（%） |
| Contribution | 對 SPY 的貢獻（%p）≈ 權重 × 報酬 |
| Share of SPY | 該貢獻 ÷ 當期 SPY 總漲跌幅 |
| Other holdings | 其餘成分與拆解誤差 |

**資料來源（Panel E/F）**

- 持股權重：SSGA 官方每日 SPY XLSX（預設，不需付費 API）
- 個股價格：`yfinance` + `data/raw/spy_constituent_prices_cache.csv`
- `FMP_API_KEY`：僅作 OpenBB 備援；免費 FMP 對 ETF holdings 會回 402

---

## 4. 資料層（`src/collect.py`）

約 **24 個指標序列** + SPY benchmark：

| 類別 | 序列（節錄） |
|------|----------------|
| FRED 通膨 | `cpi_yoy`, `core_cpi_yoy`, `pce_yoy` |
| FRED 就業 | `nfp_change`, `unemployment_rate`, `jolts_openings` |
| FRED 成長 / 利率 | `gdp_growth`, `indpro_growth`, `spread_2y10y`, `credit_spread_hy`, `fed_funds_rate` |
| 央行（regime 可選） | `ecb_deposit_rate`, `boj_policy_rate` |
| 市場日頻 | `vix`, `dxy`, `gld`, `uso`（WTI）, `xlk`–`xli`, `zq_futures` |
| 基準 | `spy` |

憑證（`.env`）：

| 變數 | 必要性 |
|------|--------|
| `GEMINI_API_KEY` | 敘事需要（可用 `--skip-llm` 略過） |
| `FRED_API_KEY` | 可選（多數序列可用 FRED 公開 CSV） |
| `FMP_API_KEY` | 可選（SPY 持股已改 SSGA 為主） |

---

## 5. 專案結構

```
run.py
src/
  collect.py          # Stage 1
  indicators.py       # Stage 2
  factor_attrib.py    # Stage 2b（A–D）
  index_contrib.py    # Stage 2c → Panel E–F
  history_match.py    # Stage 3
  regime_stats.py     # Stage 3b
  analyst.py          # Stage 4
  render.py
  config.py
  notify.py
  web.py
  futures_adjust.py
templates/daily_brief.html
tests/
docs/PROGRESS.md      # 本檔
data/raw/             # gitignored
data/processed/       # gitignored
output/               # gitignored
```

---

## 6. Phase 2 周邊功能

| 功能 | 模組 | 用法 |
|------|------|------|
| Email / Telegram | `src/notify.py` | `NOTIFY_CHANNELS` + `python run.py --notify` |
| Web UI | `src/web.py` | `python -m src.web` → http://127.0.0.1:8080 |
| 每日排程 | `scripts/daily_run.sh` | Cron 或 `scripts/com.macro.daily.plist.example` |
| CI | `.github/workflows/ci.yml` | push/PR 跑 pytest |

---

## 7. 近期重要變更（2026-05）

1. **`.env` / FMP：** `load_dotenv(override=True)`；`configure_openbb()` 注入 `fmp_api_key`；需確認編輯器已存檔。
2. **持股來源：** 預設 SSGA `holdings-daily-us-en-spy.xlsx`，避開 FMP 402。
3. **S&P 貢獻者：** 併入 SPY Return Attribution 的 Panel E / F（不再獨立大區塊）。
4. **排序：** E / F 依 Contribution 由高到低。
5. **Wikipedia fallback：** 加 User-Agent；`data/raw/spx500_tickers.csv` 快取。

---

## 8. 已知限制

1. **非即時：** 以昨收為主，不適合日內決策。
2. **Panel E/F 為近似拆解：** `權重 × 報酬` 與官方指數總回報算法不同（股息、再平衡、交易日）。
3. **Coverage：** 21 日「explained」加總可能與 SPY 總報酬不完全一致，coverage 可 >100%。
4. **少數 ticker：** SSGA 非標代碼（如 `2602335D`）yfinance 可能下載失敗。
5. **因果：** A–D 為迴歸關聯；E–F 為算術拆解；皆非因果推論。
6. **Panel A–D 排序：** 仍為固定因子順序（非依 contribution）。

---

## 9. 待辦 / 未來方向

- [ ] 同步更新 `README.md`（FMP 非必須、SSGA 持股來源）
- [ ] 13F 機構持倉（README Future layers）
- [ ] 可選：Panel A–D 依 |contribution| 排序
- [ ] 可選：官方 S&P 總回報級別的貢獻拆解

---

## 10. 日常操作

```bash
cd /path/to/analysis
source .venv/bin/activate
python run.py
open output/brief_$(date +%Y-%m-%d).html
```

```bash
# 離線 / 快速
python run.py --skip-llm

# 重拉資料與持股
python run.py --force-refresh

# 單獨測試
pytest tests/ -v
python -m src.factor_attrib
python -m src.index_contrib
```

---

## 11. 一句話總結

**MVP 與 Phase 2 核心已上線：** 每日宏觀 brief + 六面板 SPY 歸因（含 S&P 500 top movers 的因子化呈現），可單機跑通、有測試與 CI。剩餘工作以文件同步、資料邊角與長期擴充為主，不阻塞日常使用。

---

*本檔由專案開發過程整理；更新時請一併修改「報告日期」與第 7 節。*
