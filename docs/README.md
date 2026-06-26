# 交通事故重建 Pipeline

本工具從監視器影片自動重建車輛的二維行車軌跡，輸出 KML / CSV / 地圖圖片，供法鑑或事故分析使用。

---

## 目錄

1. [專案概述](#%E5%B0%88%E6%A1%88%E6%A6%82%E8%BF%B0)
2. [Pipeline 架構](#pipeline-%E6%9E%B6%E6%A7%8B)
3. [資料夾結構](#%E8%B3%87%E6%96%99%E5%A4%BE%E7%B5%90%E6%A7%8B)
4. [安裝與環境設定](#%E5%AE%89%E8%A3%9D%E8%88%87%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A)
5. [快速開始](#%E5%BF%AB%E9%80%9F%E9%96%8B%E5%A7%8B)
6. [逐步使用說明](#%E9%80%90%E6%AD%A5%E4%BD%BF%E7%94%A8%E8%AA%AA%E6%98%8E)
7. [Web 工作台](#web-%E5%B7%A5%E4%BD%9C%E5%8F%B0)
8. [設定選項與環境變數](#%E8%A8%AD%E5%AE%9A%E9%81%B8%E9%A0%85%E8%88%87%E7%92%B0%E5%A2%83%E8%AE%8A%E6%95%B8)
9. [新增場景](#%E6%96%B0%E5%A2%9E%E5%A0%B4%E6%99%AF)
10. [輸出說明](#%E8%BC%B8%E5%87%BA%E8%AA%AA%E6%98%8E)
11. [依賴套件](#%E4%BE%9D%E8%B3%B4%E5%A5%97%E4%BB%B6)
12. [已知限制與待辦事項](#%E5%B7%B2%E7%9F%A5%E9%99%90%E5%88%B6%E8%88%87%E5%BE%85%E8%BE%A6%E4%BA%8B%E9%A0%85)

---

## 專案概述

給定一段路口監視器影片，本系統會：

1. 讓使用者在影片影格上圈出感興趣的車輛
2. 以 SAM2 視訊記憶功能追蹤每輛車（不依賴 YOLO 偵測）
3. 透過使用者自訂的地面控制點（GCP）建立像素→公尺的單應矩陣（Homography）
4. 把追蹤到的地面錨點投影到真實座標系，計算速度、偵測撞擊幀
5. 輸出 KML（可疊加 Google My Maps）、CSV（每幀速度/距離）、北向地圖圖片

---

## Pipeline 架構

```
影片輸入
   │
   ▼
① 場景設定 (scene_config.py)
   │  ACCIDENT_SCENE 環境變數選擇場景
   │  SceneConfig 定義所有路徑、GPS 錨點、車輛樣式
   ▼
② GCP 校正 (calibrate_homography.py)          ← 一次性前置步驟
   │  使用者點選像素座標 + 輸入 lat/lon
   │  MAGSAC++ 擬合 → homography_calibration.json
   │  含可選的放射畸變 k1 估計
   ▼
③ 車輛框選 (Web UI → vehicle_boxes.json)
   │  在起始幀框出每輛車的 bounding box
   │  可在多個影格各框一次，追蹤器重新植入
   ▼
④ SAM2 追蹤 Stage 1 (prompt_track_accident.py)
   │  以視訊記憶模式追蹤使用者指定的車輛
   │  gate 濾波（strict / loose / off）防止遮擋漂移
   │  輸出：追蹤疊加影片 + prompt_tracks.csv
   │         (frame, vehicle, anchor_x, anchor_y)
   ▼
⑤ 投影與重建 Stage 2 (auto_reconstruct.py)
   │  像素錨點 → 單應矩陣 → 公尺座標 (east_m, north_m)
   │  shape-preserving 投影（保留曲線形狀，不被 homography 壓平）
   │  偵測撞擊幀（最近接近法 / 分離邊界法）
   │  速度視窗平滑、翻滾偵測、停止截斷
   │  道路對齊：旋轉 + 可選均勻縮放（消除魚眼壓縮）
   ▼
⑥ 輸出 (birdseye_manual_annotation.py, recognized_route.py)
      route_auto.kml          ← 道路對齊後的 KML LineString
      route_auto.csv          ← 每幀 lat/lon + 速度 km/h
      map_figure_auto.png     ← 北向地圖圖片（OSM 道路底圖）
      route_recognized.kml    ← 原始投影（無道路對齊，保留真實曲線）
      route_recognized.csv
      route_recognized.png
```

---

## 資料夾結構

```
accident_reconstruction/
├── scene_config.py                  # 場景設定與路徑管理
├── calibrate_homography.py          # GCP 校正、ViewTransformer、metric_to_latlon
├── prompt_track_accident.py         # Stage 1：SAM2 追蹤
├── auto_reconstruct.py              # Stage 2：投影 + 撞擊偵測 + 截斷
├── run_pipeline.py                  # 串接 Stage 1 + Stage 2
├── birdseye_manual_annotation.py    # 地圖輸出（KML / 圖片 / CSV）+ 道路對齊
├── recognized_route.py              # 原始投影輸出（無道路對齊）
├── manual_pre_impact_motorcycle_annotation.py  # 永康場景 legacy 手動標記
├── web_app.py                       # FastAPI Web 工作台後端
└── web_app.html                     # Web 工作台前端

data/
├── videos/                          # 原始影片 + pipeline 輸出影片（共用）
└── scenes/                          # 各場景資料（每場景一個資料夾）
    ├── _training/                   # 訓練/標註資料集（manual_motorcycle_labels）
    ├── pre_impact_motorcycle/       # 永康機車撞車場景（內建）
    ├── keelung_xinwu_yier/          # 基隆計程車撞警車場景（內建）
    └── <scene_name>/                # 動態場景（Web 工作台下載後自動建立）
        ├── scene/                   # = artifact_dir（中間工件）
        │   ├── homography_calibration.json
        │   ├── gcps.json
        │   ├── vehicle_boxes.json
        │   ├── prompt_tracks.csv
        │   └── overrides.json
        ├── <scene_name>_route_*.{png,csv,kml}   # 最終輸出（落在場景根目錄）
        └── scene.json
```

---

## 安裝與環境設定

### 需求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（建議）或 pip
- ffmpeg（YouTube 下載功能需要）

### 安裝步驟

```bash
# 在 repo 根目錄
uv sync                        # 或 pip install -e ".[dev]"
```

### 下載 SAM2 權重

```bash
# 預設使用 sam2.1_t.pt（Tiny，速度快）
# 放在 repo 根目錄
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
    -O sam2.1_t.pt
```

---

## 快速開始

### 方式一：Web 工作台（推薦）

```bash
# 啟動 Web 工作台
.venv/bin/python -m accident_reconstruction.web_app

# 開啟瀏覽器：http://127.0.0.1:8000
```

Web 工作台整合了所有步驟（① ~ ⑥），無需命令列。

### 方式二：命令列

```bash
# 使用預設場景（永康機車）
.venv/bin/python -m accident_reconstruction.run_pipeline

# 使用其他場景
ACCIDENT_SCENE=keelung_xinwu_yier \
    .venv/bin/python -m accident_reconstruction.run_pipeline
```

---

## 逐步使用說明

### 步驟一：GCP 校正（首次或重新校正時執行）

```bash
ACCIDENT_SCENE=<scene_name> \
    .venv/bin/python -m accident_reconstruction.calibrate_homography
```

- 影格視窗會開啟，**左鍵點擊**地面特徵點（斑馬線角、車道線端點）
- 在終端機輸入對應的 `lat, lon`（從 Google Maps 衛星圖讀取）
- 按 `s` 或 Enter 儲存，`u` 撤銷上一點，`q` 不存離開
- 建議至少 8 個點，且涵蓋車輛行駛的近端與遠端、左右兩側

### 步驟二：框選車輛

透過 Web 工作台（步驟 3 標記車輛）或手動建立 `vehicle_boxes.json`：

```json
{
  "objects": [
    {
      "name": "motorcycle",
      "bgr": [
        0,
        196,
        255
      ],
      "boxes": [
        {
          "frame": 80,
          "box": [
            225,
            335,
            270,
            398
          ]
        }
      ]
    },
    {
      "name": "car",
      "bgr": [
        240,
        140,
        40
      ],
      "boxes": [
        {
          "frame": 80,
          "box": [
            699,
            649,
            947,
            694
          ]
        }
      ]
    }
  ]
}
```

儲存到 `data/<scene>/scene/vehicle_boxes.json`。

### 步驟三：執行完整 Pipeline

```bash
ACCIDENT_SCENE=<scene_name> \
    .venv/bin/python -m accident_reconstruction.run_pipeline
```

或分開執行兩個 stage：

```bash
# Stage 1：SAM2 追蹤
ACCIDENT_SCENE=<scene_name> \
    .venv/bin/python -m accident_reconstruction.prompt_track_accident

# Stage 2：投影重建（重複此步驟不需重新追蹤）
ACCIDENT_SCENE=<scene_name> \
    .venv/bin/python -m accident_reconstruction.auto_reconstruct
```

---

## Web 工作台

```bash
.venv/bin/python -m accident_reconstruction.web_app [--port 8000] [--host 127.0.0.1] [--reload]
```

### 功能說明

| 步驟          | 功能                                                                                  |
| ------------- | ------------------------------------------------------------------------------------- |
| ① 影片 / 下載 | 選擇已有影片，或貼入 YouTube 網址下載（支援起迄秒數截取）                             |
| ② 校正（GCP） | 在影片影格上點選控制點，在右側地圖點選對應 GPS 位置；支援多點存檔並校正，顯示每點誤差 |
| ③ 標記車輛    | 在影格上拖曳框出每輛車（可跨多影格各框一次）；框選完後存檔                            |
| ④ 設定 + 執行 | 設定撞擊幀、車輛角色、追蹤 gate；一鍵執行完整 pipeline 並即時顯示 log                 |
| 結果檢視      | 顯示路線圖、提供 KML / CSV 下載、播放追蹤疊加影片                                     |

### UI 可調整的參數（overrides.json）

| 參數                        | 說明                                                |
| --------------------------- | --------------------------------------------------- |
| `impact_frame`              | 手動指定撞擊幀（覆蓋自動偵測）                      |
| `stop_vehicle`              | 被撞停下的車輛標籤（翻滾後截斷軌跡）                |
| `moving_vehicle`            | 持續移動的車輛（用於對齊縮放）                      |
| `gates`                     | 追蹤 gate 嚴格度：`strict`（預設）/ `loose` / `off` |
| `start_frame` / `end_frame` | 分析影格範圍                                        |
| `min_traj_speed`            | 軌跡停止繪製的速度門檻（km/h，預設 3.0）            |
| `struck_full`               | 是否保留被撞車輛的完整後撞軌跡                      |

---

## 設定選項與環境變數

| 變數               | 預設值                  | 說明                                          |
| ------------------ | ----------------------- | --------------------------------------------- |
| `ACCIDENT_SCENE`   | `pre_impact_motorcycle` | 指定使用哪個場景（對應 `SCENES` dict 的 key） |
| `WEB_PORT`         | `8000`                  | Web 工作台連接埠                              |
| `PYTHONUNBUFFERED` | -                       | 設為 `1` 讓 pipeline subprocess 即時輸出 log  |

### `calibrate_homography.py` 常數

| 常數                 | 說明                                       |
| -------------------- | ------------------------------------------ |
| `RANSAC_THRESHOLD_M` | MAGSAC++ 內點門檻（公尺），預設 2.0        |
| `MIN_GCP_SPAN_M`     | GCP 最小地面跨距（公尺），低於此值發出警告 |

### `auto_reconstruct.py` 常數

| 常數                        | 說明                              |
| --------------------------- | --------------------------------- |
| `CONTACT_THRESHOLD_M`       | 接觸判定距離（公尺），預設 3.0    |
| `SEPARATION_MARGIN_M`       | 分離判定邊界（公尺），預設 1.0    |
| `FLIP_VELOCITY_M_PER_FRAME` | 翻滾速度門檻（公尺/幀），預設 1.2 |
| `SPEED_WINDOW_SECONDS`      | 速度平滑視窗（秒），預設 0.6      |

---

## 新增場景

在 `scene_config.py` 的 `SCENES` dict 中新增一個 `SceneConfig`：

```python
MY_SCENE = SceneConfig(
    name="my_scene",
    source_video=Path("data/videos/my_clip.mp4"),
    artifact_dir=Path("data/my_scene/scene"),
    start_frame=0,
    end_frame=300,
    fps=25.0,
    # 可選的地理資訊（道路對齊需要）
    road_centerlines={
        "vehicle_a": [(lat1, lon1), (lat2, lon2), ...],
        "vehicle_b": [(lat1, lon1), (lat2, lon2), ...],
    },
    intersection_latlon=(lat, lon),
    true_impact_latlon=(lat, lon),
    stop_vehicle="vehicle_a",
    moving_vehicle="vehicle_b",
)

SCENES[MY_SCENE.name] = MY_SCENE
```

也可以直接用 Web 工作台下載影片，系統會自動建立動態場景（`scene.json`）。

---

## 輸出說明

所有輸出以場景名稱為前綴，存放在 `data/<scene_name>/` 下：

| 檔案                               | 說明                                                 |
| ---------------------------------- | ---------------------------------------------------- |
| `*_route_auto.kml`                 | 道路對齊後的路線（建議覆蓋在 Google My Maps 上查看） |
| `*_route_auto.csv`                 | 每幀 lat/lon、速度（km/h）、累積距離（m）            |
| `*_map_figure_auto.png`            | 北向地圖（OSM 底圖 + 軌跡 + 撞擊點）                 |
| `*_route_recognized.kml`           | 原始投影路線（未道路對齊，保留真實曲線形狀）         |
| `*_route_recognized.csv`           | 原始投影的每幀速度資料                               |
| `*_route_recognized.png`           | 原始投影地圖圖片                                     |
| `data/videos/*_prompt_tracked.mp4` | SAM2 追蹤疊加影片（含 mask + 速度標籤）              |

---

## 依賴套件

| 套件                  | 用途                                             |
| --------------------- | ------------------------------------------------ |
| `sam2`                | 視訊物件追蹤（Meta SAM2）                        |
| `opencv-python`       | 影片 I/O、單應矩陣計算                           |
| `numpy`               | 數值運算                                         |
| `Pillow`              | 地圖圖片渲染                                     |
| `fastapi` + `uvicorn` | Web 工作台後端                                   |
| `yt-dlp`              | YouTube 影片下載                                 |
| `supervision`         | 本 repo 核心（`BoundingBoxAnnotator`、追蹤器等） |

完整版本需求見 repo 根目錄的 `pyproject.toml`。

---

## 已知限制與待辦事項

- **魚眼鏡頭壓縮**：廣角 CCTV 使遠端車輛的投影距離被壓縮；單一放射係數 k1 可緩解但無法完全消除，需要較多 GCP 且分布廣。
- **SAM2 遮擋**：兩車重疊時 mask 可能合併；gate 濾波（strict 模式）會在 mask 異常擴大時自動重植，但嚴重遮擋下仍可能需要手動增加框選幀。
- **道路對齊精度**：OSM 中心線取自 Overpass，精度約 ±2–5 公尺；撞擊點和車輛起點需從衛星圖手動讀取。
- **單場景執行**：每次 pipeline 呼叫只處理一個場景；批次處理需透過 shell 腳本迴圈多個 `ACCIDENT_SCENE` 值。
- **多鏡頭支援**：目前假設單一固定攝影機；跨鏡頭追蹤需要額外開發。
