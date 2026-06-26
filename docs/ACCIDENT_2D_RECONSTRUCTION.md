# 車禍 2D 重建進度（pre_impact_motorcycle）

行車記錄器車禍影片 → 標註車輛軌跡 → 投影成 2D 俯視 / 經緯度，疊到 Google My Maps。
本文件記錄現場事實、處理流程、關鍵發現與取捨、最終解法、輸出檔案與重現方式。

---

## 1. 現場事實

| 項目            | 值                                                                      |
| --------------- | ----------------------------------------------------------------------- |
| 事故 GPS        | `23.0269311, 120.2497122`（台南市永康區）                               |
| 路口            | **自強路 × 高速一街二段**                                               |
| 自強路          | 走向 **西南↔東北**；西南端鑽入**國道1號（中山高速公路）下的車行地下道** |
| 高速一街二段    | 走向 **近正南北**（早期誤以為東西向，已修正）；與自強路約 **50° 斜交**  |
| 路寬            | 約 **5.6 m**                                                            |
| 機車            | 走 **高速一街二段**，由南往北接近撞擊點後停住                           |
| 汽車            | 走 **自強路**，由東北往西南，撞擊後續行進入**地下道**                   |
| 路口節點（OSM） | `23.0268405, 120.2496047`                                               |

> 道路幾何來自 OpenStreetMap，已和影片畫面（地下道在上、機車左進、汽車由下往上）核對一致。

---

## 2. 自動化 Pipeline（一鍵）

`run_pipeline.py` 把已完成的階段串成一條可重複的 pipeline：

```
(一次)  calibrate_homography.py   ->  homography_calibration.json
(輸入)  select_vehicles.py        ->  vehicle_boxes.json（使用者框出事故車輛）
階段 1  指定式追蹤（SAM2 影片記憶）  ->  prompt_tracks.csv + 標註影片（框+mask+軌跡）
階段 2  投影 + 重建 + 兩點地圖對齊   ->  對齊 KML / 北上地圖圖 / 逐影格 CSV
```

**指定車輛（輸入階段）**：跑 `select_vehicles.py`（GUI opencv / conda python）。視窗有**影格捲軸**——每台車可**捲到它出現的那一格**再框（按 `s` 框、`c` 跳過、`q` 結束），所以**車輛一開始不在畫面也沒問題**。每台車存下各自的 `frame` + box → `vehicle_boxes.json`。追蹤端 `track_vehicle` 會用各車自己的起始影格獨立追蹤再合併。`run_pipeline` 有這個檔就用它、否則用 `SceneConfig` 預設。

所有「場景專屬」設定集中在 `run_pipeline.py` 的 **`SceneConfig`**：輸入影片、影格範圍、兩個真實地圖錨點（撞擊點 + 汽車起點）；車輛框則由上面的輸入階段提供。換一段影片只要複製 config、重新校正、重框車輛即可，**階段模組不用改**。

執行（先做過一次校正後）：

```bash
.venv/bin/python -m accident_reconstruction.run_pipeline
```

| 檔案                                         | 角色                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `run_pipeline.py`                            | **入口**：`SceneConfig` + 一鍵跑階段 1→2（有 `vehicle_boxes.json` 就用使用者框）                                                     |
| `calibrate_homography.py`                    | 一次性：互動點選 GCP，配衛星經緯度，算 homography，存 `homography_calibration.json`（>4 點用 MAGSAC++/最小平方；已點過的點自動沿用） |
| `select_vehicles.py`                         | 輸入階段：在起始影格框出每台事故車 → `vehicle_boxes.json`（GUI opencv / conda python）                                               |
| `prompt_track_accident.py`                   | 階段 1：使用者每台車框一次 → **SAM2 影片記憶**追蹤；含框大小穩定閘＋不准倒退閘 → `prompt_tracks.csv`（只追指定車輛，不調偵測）       |
| `auto_reconstruct.py`                        | 階段 2：著地點 → homography 投影 → 速度/撞擊偵測 → 重建資料                                                                          |
| `birdseye_manual_annotation.py`              | 輸出層：KML/地圖圖/CSV 的繪製與**兩點對齊**（人工版與自動版共用，吃外部軌跡資料）                                                    |
| `manual_pre_impact_motorcycle_annotation.py` | 提供 homography（`VIEW_TRANSFORMER`）、速度模型、人工關鍵影格基準版                                                                  |
| `auto_track_accident.py`                     | 基準/對照：YOLO+ByteTrack 全自動偵測（汽車佳、機車漏）——已被指定式追蹤取代                                                           |

> 註：ML 階段請用 **`.venv/bin/python`**（repo 的 venv，有 ultralytics/torch/SAM2）。

---

## 3. 校正（homography）

- 流程：跑 `calibrate_homography.py`，在影片影格上點地面特徵（斑馬線角等），每點配一組從 Google 衛星圖讀到的經緯度。
- 工具特性：
    - 已點過的點用 `GroundControlPoint.pixel` 固定、**自動沿用**，加新點時只需點新的。
    - **>4 點**用 **MAGSAC++ / 最小平方**（`cv2.findHomography`）擬合；4 點用精確解。
    - 存檔會印出每點誤差與用了哪個方法。
- 目前共 **15 個控制點**，結果存於
    `data/scenes/pre_impact_motorcycle/scene/homography_calibration.json`。

---

## 4. 關鍵發現與取捨（重要）

### 4.1 魚眼天花板：homography 無法精準

這支是**廣角/魚眼**行車記錄器（畫面明顯桶形變形）。單一平面 homography 在物理上無法把整個地面投影準。實測**留一交叉驗證**：

| 方法                           | 留一誤差 (mean) |
| ------------------------------ | --------------- |
| homography                     | 4.50 m          |
| 多項式 poly2                   | 4.84 m          |
| 薄板樣條 TPS                   | 5.02 m          |
| 單視角相機標定 calibrateCamera | 失敗（發散）    |
| 徑向畸變 k1 掃描               | 4.51 m          |

全部卡在 **~4.5 m**：瓶頸是**控制點本身的噪聲＋鏡頭畸變**，不是模型。路寬只有 5.6 m → 直接投影的軌跡注定會飄出路面。**「完全照辨識曲線」＋「精確貼路面」用這支素材無法同時成立。**

### 4.2 真正的問題：地圖系統性偏移（已解）

辨識出來的**曲線形狀是對的**，不該被直線化或重建。在 My Maps 上唯一的缺陷是**整體系統性平移**（兩條線往同一方向偏）：

- 量到偏移約 **3.5 m**（往北約 3.5 m、往東約 0.8 m）。
- **不是 GCJ-02**（那會是 ~500 m，往東南）—— 是校正/基準的小偏壓。

---

## 5. 最終解法

**保留辨識曲線（不直線化、不吸附、不重建），再做兩點相似對齊把曲線擺正到真實道路上。**

- 在 `birdseye_manual_annotation.py` 的 `build_alignment()`：
    - 用兩個**使用者讀到的真實點**當錨：
        - `TRUE_IMPACT_LATLON = (23.026871, 120.249608)`（撞擊點）
        - `TRUE_CAR_START_LATLON = (23.026900, 120.249650)`（汽車起點）
    - **只旋轉、不縮放**（scale=1）：第二點為近似值，縮放會把路徑壓垮；使用者反映的是「角度錯」而非「長度錯」。
    - 結果：撞擊點精準命中（不被破壞）、汽車旋轉貼上自強路；機車很短、跟著轉但仍在高速一街二段帶內。
- 速度標籤用 **第 90 百分位**（原始 homography 有單格尖刺，例如汽車 123 → 穩健約 27 km/h）。

> 之後若換影片/重標：流程不變；要微調對齊只改 `TRUE_IMPACT_LATLON` / `TRUE_CAR_START_LATLON` 兩個真實點即可。

---

## 6. 輸出檔案

| 檔案                                                                           | 內容                                                            |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `data/scenes/pre_impact_motorcycle/pre_impact_motorcycle_map_figure.png`       | **2D 地圖視圖**（北上、辨識曲線、兩點對齊、含速度與撞擊經緯度） |
| `data/scenes/pre_impact_motorcycle/pre_impact_motorcycle_route.kml`            | **疊 My Maps 用**；辨識曲線＋兩點對齊                           |
| `data/scenes/pre_impact_motorcycle/pre_impact_motorcycle_route.csv`            | 逐影格 `frame, vehicle, lat, lon, speed_kmh, is_impact`         |
| `data/videos/pre_impact_motorcycle_birdseye_split_slow_2x.mp4`                 | 左影片＋右辨識軌跡面板（動態）                                  |
| `data/scenes/pre_impact_motorcycle/pre_impact_motorcycle_birdseye_summary.png` | 相機視角辨識軌跡靜態圖                                          |
| `data/scenes/pre_impact_motorcycle/scene/birdseye_result_sheet.jpg`            | 分割影片關鍵影格檢查表                                          |

目前估計：機車峰值約 **9 km/h**、汽車峰值約 **27 km/h**（穩健值）。

---

## 7. 已知限制

- **速度為估計值**：受魚眼投影噪聲影響，僅供量級參考。
- **地下道內無解**：在國道高架下方，衛星看不到，無法放控制點校正。
- **對齊依兩個真實點**：換場景需重新提供 `TRUE_IMPACT` / `TRUE_CAR_START`。
- **homography 本身仍不準**（~4.5 m）；最終成果靠「辨識曲線＋兩點地圖對齊」，不直接依賴 homography 的絕對位置。
