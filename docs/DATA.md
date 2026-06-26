# 資料清單與重建說明（DATA.md）

影片檔**不入庫**（檔案大、且為 YouTube 車禍影片有版權疑慮），`data/` 整個被 `.gitignore`
排除。本文件記錄每個場景的**來源網址、下載方法、本地存放路徑、校正結果**，讓協作者能在
自己的機器上把 `data/` 重建出來並驗證結果。

---

## 1. 一次性把資料重建出來

```bash
# 1) 安裝環境（會自動帶 yt-dlp）
uv sync

# 2) 依下方各場景表格下載影片，存到 data/videos/（檔名必須完全一致）
#    方法 A：用內建 Web 工作台貼網址下載
.venv/bin/python -m accident_reconstruction.web_app   # → 開 http://127.0.0.1:8000 步驟①貼網址
#    方法 B：直接用 yt-dlp（檔名見各場景）
.venv/bin/yt-dlp -f mp4 -o "data/videos/<目標檔名>.mp4" "<來源網址>"

# 3) 內建場景（pre_impact / keelung）下載完即可直接跑：
ACCIDENT_SCENE=pre_impact_motorcycle .venv/bin/python -m accident_reconstruction.run_pipeline
```

> **權重**：`sam2.1_t.pt` 由 ultralytics 於首次執行時**自動下載**，不需手動準備，也不入庫。

---

## 2. 各場景

每個場景的\*\*標記點（GCP 經緯度）\*\*存成 committed 的 `scene_records/<name>.json`（`data/` 不入庫，
這些小 JSON 才是可分享的校正成果）。下表「標記點」欄連到該 JSON，內含 `youtube_url` 與每個
GCP 的 `pixel`／`lat`／`lon`。

> 「來源網址」`<待填>` 者請補上實際 YouTube 連結——**填到對應的 `scene_records/<name>.json`
> 的 `youtube_url` 欄位**（程式是讀 JSON，不是讀本表）。
> 校正結果供**驗證重新校正是否一致**用；殘差（residual）越小越準，路寬約 5.6 m，
> 故 mean residual 控制在 ~1 m 內較理想。

> **URL → 自動校正**：見 [§4](#4-url--%E8%87%AA%E5%8B%95%E6%A0%A1%E6%AD%A3%E6%A9%9F%E5%88%B6)。下載某 YouTube 連結時，若本地有對應網址的
> 紀錄，程式會自動把標記點寫回該場景、重算單應矩陣，免去重標。

### 2.1 `pre_impact_motorcycle` — 台南永康 自強路 × 高速一街二段（機車被汽車撞）

| 項目       | 內容                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------- |
| 來源網址   | `<待填>`                                                                                  |
| 標記點     | [`scene_records/pre_impact_motorcycle.json`](../scene_records/pre_impact_motorcycle.json) |
| 下載存成   | `data/videos/pre_impact_motorcycle_source.mp4`                                            |
| 場景參數   | frames 80–180、fps 25、內建場景（`scene_config.PRE_IMPACT_MOTORCYCLE`）                   |
| 校正方法   | least-squares（MAGSAC 僅留 5/15 → 改用最小平方），無去畸變                                |
| GCP 數     | 15                                                                                        |
| 殘差       | mean **3.20 m** / max **7.08 m**（魚眼廣角，此場景殘差較大）                              |
| 原點經緯度 | 23.0268866, 120.2496444                                                                   |

### 2.2 `keelung_xinwu_yier` — 基隆 信五路 × 義二路（警車被計程車撞）

| 項目       | 內容                                                                                |
| ---------- | ----------------------------------------------------------------------------------- |
| 來源網址   | `https://m.youtube.com/watch?v=REwQUfTaDMc&ra=m`                                    |
| 標記點     | [`scene_records/keelung_xinwu_yier.json`](../scene_records/keelung_xinwu_yier.json) |
| 下載存成   | `data/videos/keelung_xinwu_yier_source.mp4`                                         |
| 場景參數   | frames 120–245、fps 29、內建場景（`scene_config.KEELUNG_XINWU_YIER`）               |
| 校正方法   | least-squares，無去畸變                                                             |
| GCP 數     | 8                                                                                   |
| 殘差       | mean **0.69 m** / max **1.96 m**                                                    |
| 原點經緯度 | 25.1341258, 121.7473854                                                             |

### 2.3 `yilan_wujie` — 宜蘭五結 無號誌路口（小貨車 × 機車）

| 項目       | 內容                                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------- |
| 來源網址   | `https://m.youtube.com/watch?v=7xQGDASAMEg`                                                    |
| 標記點     | [`scene_records/yilan_wujie.json`](../scene_records/yilan_wujie.json)                          |
| 下載存成   | `data/videos/宜蘭五結無號誌路口小貨車機車相撞　騎士彈飛半空重摔命危.mp4`（注意檔名含全形空格） |
| 場景參數   | frames 0–465、fps 30、動態場景（`data/scenes/yilan_wujie/scene.json`）                         |
| 校正方法   | MAGSAC++（8/8 inliers）＋ 去畸變 k1=0.25                                                       |
| GCP 數     | 8                                                                                              |
| 殘差       | mean **0.69 m** / max **1.76 m**                                                               |
| 原點經緯度 | 24.6781718, 121.8098346                                                                        |

### 2.4 `taoyuan_yangmei` — 桃園楊梅 高鐵南路七段（違規左轉）

| 項目       | 內容                                                                              |
| ---------- | --------------------------------------------------------------------------------- |
| 來源網址   | `<待填>`                                                                          |
| 標記點     | [`scene_records/taoyuan_yangmei.json`](../scene_records/taoyuan_yangmei.json)     |
| 下載存成   | `data/videos/【車禍影片】違規左轉的下場（2026.03.01 桃園市楊梅區高鐵南路七段.mp4` |
| 場景參數   | frames 0–199、fps 25、動態場景（`data/scenes/taoyuan_yangmei/scene.json`）        |
| 校正方法   | least-squares ＋ 去畸變 k1=-0.25                                                  |
| GCP 數     | 10                                                                                |
| 殘差       | mean **0.43 m** / max **0.77 m**（四個場景中最準）                                |
| 原點經緯度 | 24.9430254, 121.1211700                                                           |

---

## 3. 重新校正（若需要）

校正結果存在各場景的 `data/scenes/<name>/scene/{gcps.json, homography_calibration.json}`。
若這些檔不存在（例如全新 clone），用 Web 工作台重做：

```bash
.venv/bin/python -m accident_reconstruction.web_app
# 步驟② GCP 校正：左側點影片像素、右側點 OSM 街道對應經緯度，配對 ≥8 點 → 存檔
```

校正完成後比對上表的「殘差」與「原點經緯度」，數值接近即代表校正一致。
動態場景（yilan / taoyuan）的車輛框與場景參數另存於同目錄的
`vehicle_boxes.json` / `scene.json`。

---

## 4. URL → 自動校正機制

`scene_records/<name>.json` 把 **YouTube 網址** 對應到**你標記的 GCP 經緯度點**。因為這些 JSON
入庫、`data/` 不入庫，它們就是可分享、可重現的校正成果。模組：
[`accident_reconstruction/scene_records.py`](../accident_reconstruction/scene_records.py)。

**自動載入（下載時）**：在 Web 工作台貼網址下載時，程式呼叫 `find_record_by_url(url)`。
比對是用 **11 碼影片 id**，所以 `youtu.be/…`、`watch?v=…`、`embed/…` 與 `?si=`／`&t=` 等
參數都會對到同一筆。命中就把該筆 `gcps` 寫回新場景的 `gcps.json`、重算單應矩陣 —— 免重標。

**自動記錄（校正時）**：當一支已知網址的影片完成 GCP 校正，`/api/calibrate` 會呼叫
`save_record(...)` 把「網址 → 標記點」寫進 `scene_records/<name>.json`，下次就能自動載入。

```python
from accident_reconstruction.scene_records import find_record_by_url

rec = find_record_by_url("https://youtu.be/REwQUfTaDMc")
# rec["name"] == "keelung_xinwu_yier"; rec["gcps"] 為標記的經緯度點
```

> 要讓某場景被自動命中，務必把它的網址填進 `scene_records/<name>.json` 的 `youtube_url`
> （目前 `pre_impact_motorcycle` 與 `taoyuan_yangmei` 仍為 `null`，待補）。
