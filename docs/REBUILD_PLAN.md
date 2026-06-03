# Macra Times 重建計劃

## 核心原則
- **量化算「是什麼」，LLM 解釋「代表什麼」**
- **前端主題式，後端逐國量化**
- 口語輸出，縝密框架

## 目標受眾
懂宏觀概念（聽得懂解釋），不懂術語（不看 SKEW/VVIX），看 Alf/Real Vision 類型的長線散戶。

---

## 地理框架

| 國家 | 角色 | 關鍵數據 |
|------|------|------|
| US | 全球利率定錨 | CPI、NFP、Fed、殖利率曲線、信用利差 |
| Europe | 政策分歧指標 | ECB 利率、EUR/USD、歐洲 10Y 利差 |
| Japan | 套利融資來源 | BOJ 利率方向、JPY 倉位、USDJPY 動能 |
| China | 全球需求代理 | CNH、銅價、MCHI、外匯儲備（不用官方 GDP） |
| Taiwan | 資金流向目的地 | 外資流入、TWD、台股 ETF |
| EM 整體 | 資本流向目的地 | EMB、EEM、銅價 |

UK 移除。

---

## 資料決定

### 拿掉
- JOLTS（跟 NFP 重疊）
- Panel B CRR-style 歸因（跟 Panel A 重疊）
- Top SPY 個股貢獻（RSP/SPY 已覆蓋集中度風險）
- UK 國家分析

### 留
- Panel A：Tradeable Macro（黃金、殖利率、信用、油、美元、VIX）
- Panel C：FF5+Mom（簡化成 2-3 訊號：Mom、HML、SMB）
- Panel D：Sector ETFs

### 新增
- EM 整體信號：EMB、EEM、銅價（HG=F）

### 改為「異常才顯示」
- SKEW、VVIX、COT 倉位、TED spread

---

## 量化推理層（每國四格模板）

每個國家程式先算：
```
成長方向  → 加速 / 持平 / 減速（用 z-score 趨勢）
通膨方向  → 加速 / 持平 / 減速（用 YoY 斜率）
貨幣政策  → 緊縮 / 持平 / 寬鬆 + 市場預期 vs 實際的差距
資本流向  → 流入 / 流出 / 中性（FX + ETF 動能 + 流量）
```

這個結構化 JSON 才餵給 LLM。

---

## LLM 架構目標

```
現在（四個 LLM 呼叫）：
  run.py → LLM
  global_regime.py → LLM
  flow_run.py → LLM
  synthesis.py → LLM

目標（單一 LLM 入口）：
  country_signals.py → 量化，無 LLM
  flow_run.py → 量化，無 LLM
  global_regime.py → 量化，無 LLM
  run.py → 量化，無 LLM
  synthesis.py → 唯一 LLM 呼叫，做主題提煉
```

---

## 前端呈現目標

```
L3（戰略綜合）→ 2-3 個主題，各國當佐證
L2（市場動能）→ 資金流向證據
L1（宏觀重力）→ 結構性證據
```

---

## 執行階段

### 🟢 階段一：輸出層翻譯（當前）
改 synthesis.py 的 LLM prompt：術語 → 白話，主題式輸出
- 檔案：`synthesis.py`
- 風險：低，可逆

### 🟡 階段二：資料精簡
- 拿掉 JOLTS、Panel B、個股貢獻、UK
- 加 EM 整體信號（EMB、EEM、銅價）
- SKEW/VVIX/COT/TED 改成異常才顯示

### 🟠 階段三：建立量化推理層
- 建 `country_signals.py`
- 各國四格模板（純量化）

### 🔴 階段四：LLM 收斂到單一入口
- `flow_run.py`、`global_regime.py`、`run.py` 移除 LLM
- 全部改輸出結構化 JSON

### 🔴 階段五：前端改主題式
- L3 從 country tabs → theme cards
- L2/L1 重新定位成「證據層」
