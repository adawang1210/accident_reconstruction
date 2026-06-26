# 車禍事故 2D 重建 — 專案總結與進度紀錄

把行車記錄器/CCTV 車禍影片，重建成**地圖上的 2D 軌跡（含經緯度、速度、撞擊點）**，
並逐步發展成一套**可重複、多場景、可換片**的 CV pipeline ＋ **網頁工作台**。

> 技術細節另見 [ACCIDENT_2D_RECONSTRUCTION.md](ACCIDENT_2D_RECONSTRUCTION.md)。本文是**全程進度與操作紀錄**。

---

## 1. 專案目標

影片 → 偵測/追蹤事故車輛 → 投影到地面 → 在 Google/衛星地圖上畫出**真實彎曲軌跡**，
標上**經緯度、速度、撞擊點**，輸出 **KML（疊圖）** 與 **逐影格 CSV**。

---

## 2. 整體成果（現況）

- **2D 重建 pipeline（全自動）**：指定車輛 → SAM2 影片記憶追蹤 → homography 投影 → 撞擊偵測 → 兩點地圖對齊 → KML / 2D 地圖 / CSV。
- **多場景**：用 `scene_config.py` 集中設定，`ACCIDENT_SCENE` 環境變數切換。目前兩個場景：
    - `pre_impact_motorcycle`（台南永康 自強路×高速一街二段，機車被汽車撞）— 完整。
    - `keelung_xinwu_yier`（基隆 信五路×義二路，機車被計程車撞）— 已校正（11 GCP）。
- **網頁工作台**（FastAPI + Leaflet）：五步驟流程（影片→校正→標記車輛→執行→結果），可選/下載影片、點圖配對 GCP、瀏覽器內框車。
- **自動版 ≈ 手動版**：自動 pipeline 的 2D 重建與最初人工標註的結果一致。

---

## 3. 你做過的操作（紀錄）

依時間順序大致如下：

1. **提供現場資訊**：事故 GPS、路名、路寬（多次量測）。
2. **校正（GCP）**：在衛星圖讀斑馬線角的經緯度、在影片點對應像素 —— 先 4 點，後補到 **15 點**，用 MAGSAC++/最小平方算 homography。
3. **反覆疊 KML 到 Google My Maps 驗證並回饋**，促成幾個關鍵修正：
    - 指出「機車走高速一街二段、汽車走自強路、撞後進地下道」等真實路線。
    - 發現投影**歪斜/飄移**、兩車**沒交會**。
    - 點出「**要照辨識軌跡、不要直線化**」。
    - 提供「真實撞擊點」與「錯誤撞擊點」兩座標 → 讓我量出**系統性偏移 ~3.5 m**（判定**非 GCJ-02**）。
    - 提供「汽車起點」真實座標 → 做**兩點旋轉對齊**。
4. **下載第二場景影片（基隆）**、用網頁校正工具標出 **11 個 GCP** 完成校正。
5. **指定車輛**：框出事故車（機車/汽車、機車/計程車）。
6. **設計前端需求**：左影片→改為可點選影格、選影片移到步驟1、步驟2地圖用 OSM 街道、步驟3 瀏覽器內拖曳框車、流程化 stepper。

---

## 4. 技術歷程與關鍵發現

- **速度不能用像素算**：透視下同一真實速度在不同距離像素速度不同 → 必須投影到公制地面。
- **魚眼天花板**：這支是廣角/魚眼鏡頭，單一平面 homography 無法精準。實測（15 點留一交叉驗證）homography 4.5 m、TPS 5.0 m、單視角相機標定發散、徑向畸變掃描無改善 → **瓶頸是控制點噪聲＋鏡頭畸變（~4.5 m）**，路寬只有 5.6 m，所以「逐格原始軌跡」與「精確貼路面」**無法同時成立**。
- **真正卡關是地圖偏移**：辨識軌跡形狀其實對，My Maps 上是**系統性平移（~3.5 m）**——不是 GCJ-02（那會是 ~500 m）。
- **最終解法 = 保留辨識曲線 + 兩點相似（只旋轉）對齊**：用「撞擊點＋汽車起點」兩個真實點，把整條辨識曲線**旋轉**對正到真實道路（不縮放、不直線化、撞擊點精準命中）。
- **追蹤的正解 = 指定式 SAM2 影片記憶**：YOLO 偵測會漏小目標（機車）；改成「使用者框一次 → SAM2 影片記憶傳播」就穩。再加兩道閘：**框大小穩定**（剔 mask leak）、**不准倒退**（剔撞後 mask 跳回起點）。
- **同一物件可多框（重播種）**：使用者可在多個影格各框一次；每個框獨立起一段 SAM2 追蹤、依影格縫接（非「每格重丟框」那種會 leak 的 naive 做法）。用途：撞擊遮蔽只剩頭部時，在該格補一框即可重新抓回。閘門對「使用者親手框的影格」一律放行。
- **遮蔽著地點修正**：著地點原用 mask 底邊中點；當 mask 高度塌到「前一格完整車身高度」的 55% 以下（=只剩頂部可見，如安全帽），改成「框頂 + 前一格完整高度」推估真實輪子著地點，避免撞擊格著地點偏高。物體仍完整可見時不會誤觸發（SAM2 會切回整台）。
- **每車各自起始影格**：車輛不一定同時出現，改成各車從自己出現的影格獨立追蹤再合併。

---

## 5. Pipeline 架構

```
(一次)  calibrate / calibrate_web   → gcps.json + homography_calibration.json
(輸入)  select_vehicles / 網頁框車   → vehicle_boxes.json
階段1   prompt_track_accident       → SAM2 影片記憶追蹤 → prompt_tracks.csv + 標註影片
階段2   auto_reconstruct            → 投影 → 速度/撞擊 → 兩點對齊 → KML / 2D地圖 / CSV
入口    run_pipeline                → 一鍵跑階段 1→2（場景由 scene_config 決定）
```

---

## 6. 網頁工作台（web_app）

FastAPI 後端 + Leaflet 地圖，五步驟流程：

1. **影片 / 下載**：選資料夾內影片，或貼 YouTube 網址下載到指定資料夾（可填起迄秒數）。
2. **校正（GCP）**：左=可點選影格（標像素）、右=OSM 街道地圖（點經緯度，可搜尋/跳轉），配對成 GCP → 存檔校正、顯示每點誤差。
3. **標記物件**：不再固定機車/汽車，改成**新增物件**（類別：機車/摩托車/汽車/計程車/人/腳踏車）。同一物件可在多個影格各框一次（重播種），下方用**裁切縮圖**排成一排顯示、可點縮圖跳該幀、可刪單一框 → 存 `vehicle_boxes.json`（新格式 `{objects:[{name,cls,bgr,boxes:[{frame,box}]}]}`）。
4. **執行**（目前 CLI `run_pipeline`，待做成按鈕＋進度）。
5. **結果**（待把 KML 疊到地圖）。

跑法：`.venv/bin/python -m accident_reconstruction.web_app --reload` → 開 http://127.0.0.1:8000

---

## 7. 程式檔案（accident_reconstruction/）

| 角色              | 檔案                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------- |
| **設定中心**      | `scene_config.py`（多場景 + 衍生路徑）                                                    |
| **入口**          | `run_pipeline.py`                                                                         |
| **校正**          | `calibrate_homography.py`（核心）、`calibrate_web.py/html`（舊網頁版，已被 web_app 取代） |
| **指定車輛**      | `select_vehicles.py`（CLI）、web_app 步驟3（網頁）                                        |
| **階段1 追蹤**    | `prompt_track_accident.py`（SAM2 影片記憶 + 兩道閘）                                      |
| **階段2 重建**    | `auto_reconstruct.py`                                                                     |
| **輸出層**        | `birdseye_manual_annotation.py`（KML/地圖/CSV 繪製 + 兩點對齊；人工/自動共用）            |
| **人工基準**      | `manual_pre_impact_motorcycle_annotation.py`（homography、速度模型、人工關鍵影格）        |
| **網頁工作台**    | `web_app.py` + `web_app.html`                                                             |
| **基準對照**      | `auto_track_accident.py`（YOLO+ByteTrack，汽車佳/機車漏）                                 |
| **repo 原本範例** | `inference_example.py`、`ultralytics_example.py`、`youtube_ultralytics_example.py`        |
| **文件**          | `ACCIDENT_2D_RECONSTRUCTION.md`、`PROJECT_SUMMARY.md`(本檔)                               |

> ML 階段（SAM2/YOLO）用 **`.venv/bin/python`**；渲染影片/網頁校正視窗用 **conda python**（headless 版 opencv 不能讀 mp4、無 GUI）。

---

## 8. 資料夾結構（data/）

```
data/videos/                          # 所有影片集中於此（來源 + 渲染輸出）
  pre_impact_motorcycle_source.mp4          # 場景一 原始影片（輸入）
  pre_impact_motorcycle_prompt_tracked.mp4  # 追蹤影片
  pre_impact_motorcycle_*（manual/birdseye…）# 人工基準版輸出影片
  keelung_xinwu_yier_source.mp4             # 場景二 原始影片（輸入）
  keelung_xinwu_yier_prompt_tracked.mp4     # 追蹤影片
  （source_accident_crop_zoom.mp4 已刪除）

data/scenes/                          # 各場景資料（非影片工件）
  pre_impact_motorcycle/              # 場景一：台南永康
    scene/                            #   = artifact_dir（中間工件）
      homography_calibration.json     #     校正
      gcps.json                       #     控制點
      vehicle_boxes.json              #     車框
      prompt_tracks.csv               #     SAM2 追蹤
      *_result_sheet.jpg              #     檢查表
    pre_impact_motorcycle_route_auto.kml/.csv    # 自動版輸出（落在場景根目錄）
    pre_impact_motorcycle_route_recognized.*     # 原始辨識軌跡（圖/KML/CSV）
    pre_impact_motorcycle_map_figure_auto.png    # 自動版 2D 地圖
  keelung_xinwu_yier/                 # 場景二：基隆
    scene/  { homography_calibration.json, gcps.json }
  yilan_wujie/ , taoyuan_yangmei/     # 動態場景（含 scene.json）
  _training/                          # 訓練/標註資料集
```

命名規則：`<scene>_route_auto.kml` / `_map_figure_auto.png` / `_route_auto.csv` = 自動 pipeline 的成果。

---

## 9. 待辦 / 下一步

- **基隆場景跑完**：補 `true_impact`/`true_car_start`（地圖讀）→ 框車 → run_pipeline 驗證多場景。
- **清 mypy 型別債**：目前有 24 個既有 mypy 錯誤（多為 `Optional` 未收斂），mypy hook 已暫設為
    `stages: [manual]` 不擋 CI。清完後把 `stages: [manual]` 拿掉即可恢復強制檢查。
    檢查指令：`pre-commit run --hook-stage manual mypy --all-files`。
- （研究向）魚眼去畸變、單目 3D 車輛擬合、GPS ground-truth 驗證、3D 場景還原（已擱置）。

---

## 10. 本次工作紀錄（2026-06-23）

### 已完成（已改的程式）

1. **網頁 ③ 標記：固定機車/汽車 → 可「新增物件」+ 多框**

    - `web_app.html`：類別下拉（機車/摩托車/汽車/計程車/人/腳踏車）、新增物件、物件 chip、同一物件多影格各框一次、下方裁切縮圖排（點跳幀、可刪單框）。存檔新格式 `{objects:[{id,cls,name,label,bgr,boxes:[{frame,box}]}]}`，載入相容舊扁平格式。
    - `web_app.py`：新增 `GET /api/crop`（裁切縮圖 JPEG）；`save_vehicles` 計數支援新格式。

2. **多框餵進 SAM2（重播種+縫接）**

    - `prompt_track_accident.py`：新增 `anchor_boxes()`、`_segment_masks()`；重寫 `track_vehicle`——每個使用者框各起一段獨立 SAM2、依影格縫接；兩道閘對「使用者親手框的影格」放行。單框=一段=原行為。
    - `run_pipeline.py`：`load_init_vehicles()` 讀新/舊格式，把整串 `boxes` 往下傳（取最早幀當初始 prompt）。
    - 驗證：兩段整合測試通過，predictor 重用會經 `on_predict_start→init_state` 重置 state。

3. **網頁 ④ 執行 + ⑤ 結果**

    - `web_app.py`：`_start_pipeline_job`（子行程跑 `run_pipeline`、帶 `ACCIDENT_SCENE`）、`POST /api/run`、`GET /api/run/status`（串流 log + 完成回傳 results）、`GET /api/result`（figure/kml/csv/tracked）。
    - `web_app.html`：④ 執行鈕 + 即時 log + 輪詢；⑤ figure PNG + 追蹤影片內嵌、右側 Leaflet 用 CSV 畫各車軌跡折線+起點+撞擊點、KML/CSV 下載。
    - 驗證：執行可啟動、log 串流、結果端點皆 200。**Stage 1 追蹤成功**（car 8、motorcycle 75 幀）。

4. **修 typo `motocycle`→`motorcycle`**（`scene_config.py`、`select_vehicles.py`、`manual_pre_impact_motorcycle_annotation.py`）——讓場景 label 與 UI 類別 key 一致。

5. **`write_map_figure` 容錯**（[birdseye_manual_annotation.py:679](../accident_reconstruction/birdseye_manual_annotation.py) `if label not in aligned: continue`）——加場景無 geo 的物件不再崩。

6. **GCP 真實跨度警示**：`build_calibration` 算 `target_span_m`（target 公制 bounding box 對角線），< `MIN_GCP_SPAN_M`（15 m）時回 `span_warning`；CLI 印、網頁 `/api/calibrate` 回傳、HTML 校正結果紅字顯示。實測 keelung 9.6 m 觸發、pre_impact 29.4 m 靜默。

### 撞擊偵測規則修正（2026-06-24）

`detect_impact` 從「第一個 \<3 m」改為「**最接近點 + 撞後明顯分開（≥ `SEPARATION_MARGIN_M`=1 m）才採用，否則退回第一個 \<3 m**」，並改成**場景無關**（任意車數/車名，\<2 車回 None，>2 車取最接近的一對）。原因：壓縮座標下兩車一出現就 \<3 m → 舊規則太早觸發、把 stop_vehicle 截到剩 1 幀。

- keelung：撞擊 140→**158**，警車 1→**15 幀**（畫得出來）。
- pre_impact：撞後黏在 ~2 m 平台、不分開 → 退回第一個 \<3 m，仍 **105**（不變，無回歸）。

### 路線畫到道路外 → 道路約束投影（2026-06-25）

使用者回饋:路線仍被畫到非道路(店家街區)上、方向錯。量測證實 keelung 計程車離義二路中心線**平均 5.4m/最大 9.5m**、警車離信五路**最大 17.9m**(路寬僅 5.6m)——投影+對齊啟發法(true_start 相似、旋轉)擺不準,2D 噪聲讓路徑漂離道路。

- **取捨**:使用者明確「在路上 > 保留辨識曲線」。改用 birdseye 既有但被棄用的**道路約束**(`geo_anchored_metric` 思路)。
- **`_aligned_latlon` 重寫**:有 OSM 道路中心線 + true_start 的車,位置一律取自**道路中心線**(用 `_road_metric`/`_point_at_arc`),沿路 arc-length 由「true_start→true_impact」內插、進度來自辨識的累積行駛距離(穩健的 1D);沒有道路/true_start 的車退回旋轉對齊(`build_alignment`)。`write_kml` 也改用 `_aligned_latlon`。
- 結果:keelung 兩車離中心線 **0.00m**(完全在路上),計程車沿義二路、警車沿信五路往西北(=被撞進店裡方向)。pre_impact 無 true_vehicle_starts → fallback,**不回歸**。
- 取捨說明:這放棄了「辨識曲線細節」,改取「道路真實形狀」;因 2D 投影噪聲(~4m)遠大於路寬,辨識側向其實是噪聲,道路約束更可信。

### 2D 路線被壓平 → 保形投影（2026-06-25）

使用者實際看影片判斷路徑該有曲線,但 2D 圖被畫得太直。**證實**:計程車去畸變後辨識曲度 0.207,但 homography 壓成 **0.019**——因為曲線(~1.5m 偏離)小於校正噪聲(~4m),全域投影把它抹平。

- 上網查:**TPS** 文獻用於魚眼地面投影,但實測會**過彎(0.27–0.45)+ 外插發散(警車弦長爆 15m)**,放大 GCP 噪聲 → 排除(印證記憶「TPS 5.0m」)。
- 正解 = **保形相似投影**(`shape_preserving_metric`):homography 只決定位置/大小/方位;per-vehicle 從「去畸變像素路徑」擬合一個\*\*相似變換(旋轉+縮放,不剪切/不透視)\*\*到 homography metric,套回去畸變路徑 → **保留辨識曲線形狀**(計程車回到 0.193)、又不發散。speed/撞擊仍用 homography metric(準)。
- **gate**:僅在有 `distortion`(乾淨去畸變路徑)時套用;無畸變模型(pre_impact)會把原始像素抖動當曲線→鋸齒,故退回 homography(pre_impact 曲度 0.099,**無回歸**)。
- 與既有 `true_vehicle_starts` 兩點相似(拉伸到真實距離)疊加 → keelung 最終:**曲線保留 + 距離正確**。

### 魚眼去畸變整合 + 被撞車完整推進（2026-06-25）

**① 魚眼去畸變(解決校正天花板)**:keelung 平面 homography 的 LOO 交叉驗證殘差高達 **26 m**(in-sample 1.4m 是過擬合假象、MAGSAC 0/12)。加**單一徑向 k1** 去畸變後 LOO→**4.3m**、MAGSAC **8/12**、k1=-0.8。

- `calibrate_homography`:`undistort_to_normalized()`、`estimate_radial_k1()`(掃描 k1 取 LOO 最小);`build_calibration(gcps, image_size)` 在去畸變座標擬合、存 `distortion:{k1,cx,cy,f}`。
- 投影:`ViewTransformer(distortion=)` 投影前先去畸變;`_load_gps_calibration` 讀 `distortion`。web/CLI 校正都帶入影像尺寸。舊校正(無 distortion)維持原行為(向後相容,pre_impact 不變)。
- **重要認知**:去畸變讓距離變**準=更短**(計程車 11.9→7.7m、警車完整 9.5→3.9m),不是變長。校正本身從「無效」變「可信」。

**② 被撞車完整推進**:override `struck_full`(UI「被撞車路徑：顯示完整推進」)→ `build_data` 不在撞擊點截斷被撞車,顯示其完整在地軌跡(真正翻滾仍由 `flip_onset` 移除)。keelung 警車 7→60 幀。

- ⚠️ 但目前 strict 重追蹤已把警車「撞進店裡後的靜止段(f205–245)」當倒退丟掉,所以完整也只到 f204(~3.9m)。要看到**整段推進到店裡**,需再 gates=off + struck_full + 完整重跑追蹤。

### 結果頁改成儀表板（2026-06-25）

步驟 ⑤ 從「狀態列＋圖片/影片直疊」改成**報告式儀表板**(參考 dashboard/report 設計):

- **摘要數字卡**:撞擊影格、車輛數、分析影格範圍。
- **各車圖例卡**:色條＋顯示名＋(內部名·點數·軌跡長 m)＋最高/平均速度;顏色取自場景(與右側地圖折線、2D 圖一致)。
- **圖/影片分頁**(2D 重建圖 / 追蹤影片),取代直疊;下載 KML/CSV 鈕美化;無結果時顯示空狀態。
- 右側 Leaflet 地圖仍為互動主角(同色折線＋白邊起點＋撞擊點)。
- 數據(速度/路徑長/撞擊幀)從 route CSV 即時計算。

### 各車對齊到自己道路（per-vehicle road alignment，2026-06-25）

舊的兩點對齊只把**移動車**那條對正到真實道路,被撞車(垂直路)只跟著同一旋轉 → 方向不合邏輯。改成**每台車各自旋轉**:以該車「撞擊幀位置」為樞紐 → 真實撞擊點,旋轉到該車**自己 `road_centerlines` 在路口附近的方位**(用 PCA 取辨識路徑主方向、再對齊道路切線並用行進方向定正負)。保留辨識曲線形狀、不縮放。**通用**:任何有道路中心線的場景皆適用,不需逐場景手調錨點。

- 驗證(對齊後路徑方位 vs 道路方位):keelung 計程車 0.3°、警車 0.0°;pre_impact 汽車 0.3°、機車 0.3° —— **每車都 < 0.5° 貼齊自己道路**。
- `build_alignment` 改回傳 `align(latlon, label)`(per-vehicle);撞擊點直接用 `TRUE_IMPACT_LATLON`(每車撞擊幀位置都映到它)。
- `is_geo_ready` 不再強制 `true_car_start_latlon`(各車靠自己道路定向,少一個手動錨點);`moving_vehicle`/`TRUE_CAR_START` 對齊上已不用(留著無害)。
- 圖標題改「各車對齊到自己道路」。

### 被撞車「翻滾」軌跡截斷（2026-06-25）

被撞車(keelung 警車)被撞飛/翻滾後,著地點 anchor(mask 底邊中點)失去物理意義 → 軌跡亂跳、疊圖不合邏輯。資料證實警車軌跡分三段:f140–198 平順(接近+被推,~0.5 m/幀)、f199–215 翻滾(突跳 1.5 m、來回彈)、f216+ 停住抖動。

- **`auto_reconstruct.flip_onset()`**:撞擊後第一個「每幀位移 > `FLIP_VELOCITY_M_PER_FRAME`(1.2 m/幀)」= 離地翻滾起點。`build_data` 用它截斷 stop_vehicle,**只丟翻滾、保留乾淨的接近+被推段**;無翻滾時退回撞擊截斷(pre_impact 機車仍 85–105,無回歸)。keelung 警車 11→**59 幀(140–198)**,地圖翻滾迴圈消失。
- **追蹤影片軌跡線**:`prompt_track` 讓 stop_vehicle 的連線在「離起點最遠那幀(≈翻滾起點)」停止延伸(框/點仍逐幀畫,CSV 仍完整,由 reconstruct 做 metric 截斷)。
- **修正 gates 語意**:strict = 正常行駛/翻滾車(翻滾另由截斷處理);loose 僅限「平順倒車/退後」(UI 標籤與註解都改)。
- 仍存:警車地圖**方向**未完全貼齊信五路(兩點對齊只對正計程車那條;要各車各自貼道路中心線是另一較大改動)。

### 前端設定化 + override 機制（2026-06-24）

把「不好寫死、每支影片不同」的設定改成 **UI 可選 + 每場景 `overrides.json`**（artifact_dir，UI 寫、stage 讀，凌駕 scene_config 預設）：

- **撞擊幀**：UI 可選「自動 / 手動指定」→ `build_data` 優先用 `SCENE.impact_frame_override`。實測 170→27 幀、164→21 幀、清除→自動 158→15 幀。
- **被撞/停下車（`stop_vehicle`）、移動/對齊車（`moving_vehicle`）**：UI 下拉（車輛名來自 `vehicle_boxes.json`）→ `resolved_stop_vehicle` / `resolved_moving_vehicle`。
- **追蹤閘（`gates`）**：UI 選 strict（預設）/ loose（撞飛/翻車：放寬 size、關 no-backtrack）/ off。`track_vehicle` 依 `SCENE.gate_mode` 套用。
- **新端點**：`/api/overrides`（讀寫）、`/api/reconstruct`（只重算 stage 2，秒級，給改設定後快速迭代）。UI 步驟 ④ 加設定列 + 「重算結果（不重追蹤）」鈕。
- **修正**：`ROAD_WIDTH_M` → `scene_config.road_width_m`（道路繪製寬度，不再寫死永康 5.6）；`CJK_FONT_PATH` 加多路徑 fallback + `_font` 退回預設字型（離開 mac 不崩）。
- ⚠️ **網頁 server 需重啟**才會載入新端點/UI（目前是 `python -m accident_reconstruction.web_app`，無 `--reload`）。

### keelung 最終驗證（2026-06-24）✓

撞擊 **158**、計程車 126 幀、**警車 1→15 幀**（地圖兩車都畫出、KML 含計程車/警車/撞擊點）。計程車仍 ~10 m 是魚眼投影天花板（非此次問題）。

### 🟡 寫死設定盤點（2026-06-24）— 每支影片情況不同（部分已 UI 化）

**已 scene 化（正確隨影片變）**：車輛與框（任意數量/類別）、`stop_vehicle`（被撞/停下車）、`moving_vehicle`（對齊錨點車）、道路/路口/真實錨點、fps、幀窗、weights、撞擊偵測（場景無關）。

**仍寫死（不同影片的風險）**：

1. **no-backtrack 閘**（`BACKTRACK_TOLERANCE_PX`，prompt_track）——假設車「永遠不往起點倒退」。撞後被撞飛/倒車的車會被誤剔。全域恆開、非 scene 可調。
2. **size-stability 閘**（`SIZE_RATIO` 0.6–1.7）——假設框大小漸變。翻車/翻滾（keelung 警車就翻覆）長寬劇變會被剔（keelung 因 stop_vehicle 截斷影響不大，但通則有風險）。
3. **`ROAD_WIDTH_M = 5.6`**（manual 模組，寫死永康路寬）——keelung 路寬不同卻畫同寬（僅示意圖外觀）。應移到 scene_config。
4. **`CJK_FONT_PATH`**（寫死 macOS 字型路徑）——缺字型/非 mac 會崩，可攜性問題。
5. **`CONTACT_THRESHOLD_M`/`SEPARATION_MARGIN_M`**、地圖版面比例（`scale=13`、`size=880`、`BIRDSEYE_PX_PER_M=26`）——固定常數，超大場景圖可能裁切。
6. **結構味道**：`auto_reconstruct` 從 `manual_pre_impact_motorcycle_annotation` import `VIEW_TRANSFORMER`（執行期 scene-aware，但命名/位置誤導）。

### 🔴 keelung 根因（重要發現，2026-06-23）

基隆地圖輸出壞掉的根因是**校正不是追蹤**：11 個 GCP 像素橫跨整個畫面，但真實經緯度只散 ~9.6 m（橫向僅 ~6.2 m，路口實際橫越 ~30 m）→ 距離壓縮 3–4 倍 → 計程車軌跡偏短、撞擊誤判在 f140（實際 ~164）、警車被截到剩 1 幀沒畫出。重投影誤差 0.70 m 是假象。**追蹤本身（多框重播種）是好的，影片證實兩車分割乾淨。修法 = 回 step ② 用真實位置拉開的 GCP 重新校正**（近端+遠端、最左+最右），不是改程式。span 警示就是用來事前擋下這類校正。

### 未完成 / 卡點（下次接續）

- **🔴 `write_map_figure` 容錯（最優先，改到一半被打斷）**：上次整跑在 stage 2 崩 `KeyError`（[birdseye_manual_annotation.py:671](../accident_reconstruction/birdseye_manual_annotation.py) `frames = sorted(aligned[label])`）。typo 已修正可解此場景，但仍要加 `if label not in aligned: continue`，否則日後加「人/腳踏車/第二台車」等場景沒 geo 的物件會再崩。`write_kml`/`write_csv` 已有容錯，只剩 `write_map_figure`。
- **🟠 修完要重跑整條 pipeline 驗證**：確認 motorcycle 能重建、figure/KML/CSV 重新產出、地圖兩條軌跡都出來。
- **🟠 遮蔽著地點修正（原本答應的功能，尚未實作）**：撞擊遮蔽時 anchor=可見 mask 底邊（帽緣）會偏高。計畫用「前一格完整車身高度」推估著地點（`OCCLUSION_HEIGHT_RATIO` + `occlusion_corrected_anchor` helper），目前只在 docstring/§4 標 TODO，程式未加。
- **🟡 car 只追到 8 幀**：car 多框（80/90/100）追蹤結果偏少，待查是閘門剔除還是 SAM2 跟丟。
- **🟡 物件名 ↔ 場景 geo 角色對應**：geo 階段用場景車名當 key，UI 用類別 key；目前靠「使用者選的類別剛好＝場景 label」才對得上。加不相符物件無道路對齊（且需上面的容錯才不崩）。未來可做「把物件指派到場景道路/角色」的 UI。
- **🟡 超窗框靜默略過**：car 有 frame 200 的框但窗是 80–180，會被略過（不出錯），未來可在 UI 提示。

---

## 11. 後續修正與前端強化（2026-06-23 續）

### 已完成（本輪）

1. **右半邊改成 Google 地圖**（`web_app.html`）

    - Leaflet 圖磚改用 Google：預設「地圖」(`lyrs=m`，含店家標籤)＋「衛星 (`lyrs=s`) / 混合 (`lyrs=y`)」切換；`subdomains mt0–3`。點地圖照樣回傳經緯度、不需 API key（非官方 API，ToS 注意；正式可改官方 Maps API + 金鑰）。

2. **左半邊選影片 + YouTube 下載**（先前已具備，本輪確認可用）

    - `web_app.py`：`GET /api/videos`（列 `data/` 下可選影片）、`POST /api/download`（yt-dlp 下載到指定資料夾、可指定起訖秒）、`GET /api/frame`、`GET /media/{path}`（含 Range 可拖曳）。

3. **🟢 下載/選取的影片自動成為可用場景（動態場景）**

    - `scene_config.py`：`SceneConfig.to_dict/from_dict`、`discover_scenes()`（匯入時掃描 `data/**/scene.json` 註冊；內建場景不被覆蓋）。
    - `web_app.py`：`_register_dynamic_scene()`——沒有內建場景的影片，依資料夾名建立 SceneConfig、加入 `SCENES`、寫出 `data/<folder>/scene.json`，讓 `run_pipeline` 子行程能用 `ACCIDENT_SCENE` 載入。`_scene_for_video()` 找不到內建場景時自動建動態場景。
    - 限制：地圖圖/KML 等「地理錨定輸出」仍需該場景的道路中心線 + 真實錨點（`is_geo_ready` 控管）；校正/追蹤/原始投影可用。

4. **🔴 `write_map_figure` KeyError 容錯（已修）**

    - `birdseye_manual_annotation.py`：`write_map_figure` 加 `if label not in aligned: continue`；`_panel_path` 改 `metric.get(label, {})`。資料缺某物件不再崩，正常出圖。

5. **🟡 car 只追到 8 幀（已修，根因＋加速）**

    - 根因：多框把單一物件切成互不重疊的短段，每段都是「無影片記憶的全新 SAM2 re-seed」很快跟丟；且 SAM2 跑在 **CPU**。
    - `prompt_track_accident.py`：
        - 每個框都傳播到 `end_frame`，合併時「最近一次仍抓得到的 re-seed 優先，失效回退第一框的連續骨幹」→ 加框只會更好、不會縮短。
        - `_box_near()`：早段已追好的**冗餘框自動跳過**（不重跑）→ 乾淨影片 N 個冗餘框 ≈ 一條連續軌跡的成本。
        - `_select_device()`：SAM2 改跑 **MPS**（Apple GPU）。
        - `_segment_masks()`：跟丟超過 `LOST_PATIENCE_FRAMES=20` 幀**早停**。
    - 驗證（永康重跑）：**car 8→96 幀（80–180）**、機車撞擊前路徑乾淨、**撞擊影格 86→105（正確）**；地圖圖/KML/CSV 重新產出正常。

6. **基隆場景 geo 資料**：用 Overpass 查得 信五路/義二路中心線 + 路口 (25.1341019, 121.7474411)，填入 `scene_config.KEELUNG_XINWU_YIER`（仍待 `true_impact`/`true_car_start` + 框車）。

7. **校正工具修正**：`calibrate_web.py`、`select_vehicles.py` 等補 `sys.path`（用 `python 檔案.py` 直接執行也能 `import examples`）。

8. **Lint**：`pyproject.toml` 對 `web_app.py`/`calibrate_web.py` 加 `RUF001/2/3` per-file-ignore（中文 UI 全形標點為刻意）；清掉 `web_app.py` 既有 E501。全部 ruff 通過。

### §10 卡點更新

- 🔴 `write_map_figure` 容錯 → **✅ 已修**（見上 4）。
- 🟡 car 只追到 8 幀 → **✅ 已修**（見上 5）。
- 🟠 遮蔽著地點修正 → 仍未實作（`occlusion_corrected_anchor` 已在，但著地點偏高的整體修正待驗）。
- 🟡 物件名↔場景 geo 角色對應、🟡 超窗框靜默略過 → 仍待 UI 強化。

### 仍待辦

- 基隆：地圖讀 `true_impact`/`true_car_start` → 框車 → `ACCIDENT_SCENE=keelung_xinwu_yier` 跑 `run_pipeline` 驗證多場景。
- 重要操作提醒：(a) `web_app` 改 `.py` 後要**重啟伺服器**（HTML 改即時生效）；(b) **勿同時跑兩條 pipeline**（會寫同一 CSV 互相汙染又佔 GPU）。

### 追加修正：機車軌跡撞擊後回跳起點（已修）

- 症狀：追蹤影片中,機車軌跡在撞擊後會連一條線回到原始起點。
- 根因:使用者在 frame 140 畫了機車修正框(實際在前方),但 SAM2 從該框 re-seed 時回跳到原始 prompt 位置(=起點);因 140 是「使用者親手框的影格」,原本**同時跳過尺寸閘與不准倒退閘**,壞點被保留。
- 修法([prompt_track_accident.py](../accident_reconstruction/prompt_track_accident.py)):**不准倒退閘改為對所有影格生效**(含使用者框);僅尺寸穩定閘仍為遮蔽放行使用者框。
- 驗證:重跑後機車止於 f139 (688,446),不再出現 f140→(228,387);影片中回跳線消失。

---

## 12. 基隆場景驗證 + 車輛身分更正(2026-06-24)

### 🔧 重大更正:基隆事故車輛 ≠「機車被計程車撞」

- 逐格看完 `data/videos/keelung_xinwu_yier_source.mp4`(696 幀、29fps、1280×720),實際事故是:
    **黃色計程車(左→右穿越路口)正面 T 撞上一台警車,把警車撞翻**(f180 起翻覆,f240 後側翻在右側 實德中藥行 前)。**全程沒有機車**。
    (另有一台白色警車從畫面下方進場 = 警匪追逐;被撞翻的就是它。先前一度誤判成「紅色休旅車」,其實是翻覆時被尾燈/底盤照成偏紅的警車。)
- **撞擊 ≈ f164–168**(00:55:56)。計程車最早完整可見 f130(左側),警車撞前 f148(畫面下方、藍色警示燈條)。
- 使用者確認:要重建的兩台車 = **計程車(移動/撞擊方) + 警車(被撞翻,= stop_vehicle)**。

### 已完成(本輪改的程式)

1. **`scene_config.py` 重寫 `KEELUNG_XINWU_YIER`**:`motorcycle`→`police_car`;`stop_vehicle="police_car"`;新增 `moving_vehicle="taxi"`;window 改 `120–210`(聚焦接近→撞擊);填入 `init_vehicles`(taxi@130、police_car@148)、`vehicle_display`(計程車/警車)、`road_names`(taxi 義二路、police_car 信五路,**道路對應仍待地圖驗證**)、`road_centerlines`(police_car 繼承原信五路線)。
2. **新增 `SceneConfig.moving_vehicle` 欄位**(+ to_dict/from_dict;永康補 `moving_vehicle="car"`)。
3. **修硬編 label**:`birdseye_manual_annotation.py` 的 `build_alignment` 原寫死 `metric["car"]` 當兩點對齊的「移動車起點」→ 改用 `MOVING_VEHICLE = SCENE.moving_vehicle or "car"`(基隆移動車是 taxi,否則對齊會壞)。其餘 stage 皆已用 `SCENE.stop_vehicle`、`detect_impact`/`_impact_point` 與 label 無關。
4. **兩個真實錨點用 homography 投影估計**(基隆校正殘差 ~0.7 m,比永康魚眼可信):`true_impact_latlon=(25.1340898,121.7474454)`(撞擊點 ~f164 地面點)、`true_car_start_latlon=(25.1341163,121.7474077)`(計程車起點 ~f130)。**待使用者在地圖上微調**。
5. **`vehicle_boxes.json`**:寫入 taxi@130 `[12,205,238,298]`、police_car@148 `[268,315,562,478]`。
6. **網頁小改**:`/api/scene` 回傳 `start_frame/end_frame`;前端載入場景時把標記影格預設到 `start_frame`(基隆=120,否則從 0 看不到車),`sceneTag` 顯示影格範圍。

### 首次整跑結果(end-to-end 成功,但追蹤品質待修)

- `ACCIDENT_SCENE=keelung_xinwu_yier run_pipeline` 跑通:KML/figure/CSV/tracked 全產出,**多場景框架驗證 OK**(設定/校正/投影/兩點對齊/道路幾何皆正確,地圖圖的 信五路×義二路 X 形正確)。
- **但兩台車軌跡塌成一點**,根因是撞擊處 SAM2 跟丟/身分混淆:
    - **計程車 anchor 從 ~f150 起凍結在 (105,264)**——被警車/碎片遮蔽後跟丟,軌跡凍在左側。
    - **警車 mask 反而一路往右(431→982 px)**,疑似撞擊後 latch 到計程車;且 `police_car` 為 stop_vehicle 被截到 impact(偵測為 f149)→ 只剩 2 幀。
    - `detect_impact` 因計程車凍結 + 投影噪聲,把撞擊偵在 **f149(偏早,實際 ~164)**。

### 下一步(使用者將自行在前端補多框)

- **伺服器已啟動**:`.venv/bin/python -m accident_reconstruction.web_app`(背景, http://127.0.0.1:8000;log 在 `/tmp/webapp.log`)。注意 `/api/run` 用 `sys.executable`,故**必須用 `.venv` 啟動**子行程才有 SAM2。
- 前端 step③:挑 `data/videos/keelung_xinwu_yier_source.mp4`(自動選到內建場景、標記影格預設 120),計程車/警車已各有一框。**計程車需在 ~f160–200 補多框**(它跨越路口往右、撞後仍前進),讓 SAM2 重播種接回;警車為 stop_vehicle 只需撞前位置即可。
- 補完框 → step④ 執行 → step⑤ 看 KML/軌跡;再回頭微調兩個真實錨點。
- ⚠️ 改 `.py` 需**重啟伺服器**;**勿同時跑兩條 pipeline**(共寫 prompt_tracks.csv + 佔 GPU)。

### 現況更新(2026-06-25)— 多框已補、校正已改善,但 KML 仍壓縮(魚眼天花板)

- **追蹤已修好**:使用者前端補多框後,計程車 126 點全程橫越乾淨、警車 60 點,影片證實兩車分割正確。window 已延伸到 `120–245`(原 210 會丟掉 f240 的計程車框)。
- **校正已重做**:keelung 現 12 GCP + 魚眼畸變 `k1=-0.8`,真實跨度 6 m→**20.6 m**(過 15 m 門檻,span 警示不再觸發)。
- **但 2D/KML 仍不符實際**,問題**從「外插壓縮」轉成「魚眼單平面 homography 天花板」**:
    - 計程車像素橫跨整個畫面(47→1122 px),投影後卻只落在校正區**內部**的 ~5×4 m(start_m≈(-0.7,-1.5)→end_m≈(4.0,2.1)),全程僅 **7.7 m**(警車 3.9 m);真實橫越約 20–30 m → 仍壓縮 ~3×。
    - 對齊是**逐車 pivot 於撞擊點 + 只旋轉對齊各自道路、不縮放**(`build_alignment` 已改 per-vehicle),保留辨識長度 → KML 疊到 Google Map 就是兩條 ~8 m / ~4 m 短截線,位置/長度不對。
    - 根因 = 一張平面 homography 無法在廣角魚眼整個視野保持距離(near/far 壓縮),與永康同一天花板;GCP 已不在外插區,改善 GCP 也救不回這段。
- **方向已定(使用者選 B)＋已實作**:對齊加入**逐車 scale**。
    - `scene_config.py`:新增 `true_vehicle_starts: dict | None`(`{label:(lat,lon)}`,+ to_dict/from_dict)。keelung 填入使用者地圖讀的真實起點:taxi `(25.1340989,121.7474192)`、police_car `(25.1340959,121.7474648)`。
    - `birdseye_manual_annotation.build_alignment`:有 `true_vehicle_starts[label]` 的車改用**兩點相似(撞擊+起點,含 scale)**——撞擊→TRUE_IMPACT、起點→真實起點,scale=真實距離/辨識距離,bearing 由兩點決定(覆蓋道路 bearing);沒設的車維持「只旋轉、貼道路」。transforms 由 2-tuple 改 3-tuple 帶 scale。
    - 結果:計程車路徑 **7.7→19.3 m**(scale≈2.47)、警車 **3.9→8.7 m**(scale≈2.26);KML 計程車從真實起點沿義二路展開。**待使用者把 KML 疊 Google My Maps 驗證**;若仍偏短/偏向,調 `true_vehicle_starts` 兩點即可(起點也決定方向,需沿該車道路精準點)。
    - 注意:scale 由「撞擊→起點」短段(辨識僅 ~1 m)推得 → 敏感;魚眼壓縮非均勻 → 端點準、中段可能微彎。(C) 深層解(相機標定/單目 3D)仍擱置。

### 錨點 UI 化(step ② 可輸入撞擊點+每車出現點,2026-06-25)

- **動機**:讓 2D 對齊的地理錨點變成前端可調,使用者點地圖即可改、重算看 2D 圖是否更準(不必改程式)。
- **後端**:
    - `scene_config`:新增 `resolved_true_impact_latlon` / `resolved_true_vehicle_starts`(override 優先,合併 scene 預設)。撞擊點預設更新為使用者讀的 `(25.1341166, 121.7474306)`。
    - `birdseye`:`TRUE_IMPACT_LATLON`/`TRUE_VEHICLE_STARTS` 改讀 resolved\_\*(吃 overrides.json)。
    - `web_app`:新增 `POST /api/anchors`(**只 merge 地理錨點進 overrides.json,不動 step④ 的 impact_frame/roles/gates**);`/api/scene` 回傳 `true_impact`/`true_vehicle_starts` 現值。`save_overrides` 也保留既有錨點 key。
- **前端**:step ② 校正下方加「🎯 2D 對齊錨點」區——撞擊點 + 每車「出現點」lat/lon 欄位,右圖點地圖後按「填入點選」帶座標,「儲存錨點」存檔。回 step④ 執行即用新值重算。
- **驗證**:端點 round-trip OK(merge 不洗 step④);新撞擊點重算 → 計程車 15.5 m、警車 17.3 m(警車 scale 隨新撞擊點放大,因 struck_full 保留全程 + 短基線敏感)。my 改的檔 ruff 全過(`calibrate_homography.py` 補進 RUF001/2/3 ignore,因新增中文警示字串)。

### 前端動效優化(LottieFiles motion-design-skill,2026-06-25)

- **下載 skill**:`git clone` 到 `skills/motion-design-skill/`(原則型 markdown 指南,非執行檔;Corporate/Premium 動效人格、時長/緩動表、Disney 原則、stagger 等)。`.claude/launch.json` 新增 `web_app` 設定(供 preview 工具啟動截圖)。
- **套用到 `web_app.html`(純 CSS + 少量 JS,Corporate 人格)**:建立動效 token(招牌曲線 `--ease`、三段時長 `--dur-quick/dur/dur-slow`、進場緩動 `--ease-emph`)。
    - 切步**面板進場**(fade + 10px rise);**按鈕** hover 上浮 / press squash(0.97);**步驟 chip** 號碼 active 放大 pop + 平滑狀態轉換;**step⑤ 儀表板**指標卡→車輛卡 **stagger 進場**(40→340ms,< 500ms wave budget);**GCP 點 pop-in**;thumb/chip 微 cascade;**校正/錨點存檔成功閃綠**(`flashOk`,state-feedback)。
    - **無障礙**:`@media (prefers-reduced-motion: reduce)` 全面停動效(skill 的 context-adaptation 要求)。
    - 遵守 1/3 規則(不同時動全部)、entrance 用 decelerate 緩動。preview 截圖確認無破版;HTML 每請求即時讀取免重啟。

### 速度門檻軌跡截斷(車輛停住就停線,2026-06-25)

- **問題**:警車被撞後在 ~f178 停住,但 SAM2 續追到 f204,停住後的 anchor 抖動畫出一段無意義「鉤」(使用者在 Google 圖上看到)。原 `flip_onset` 截斷在此**沒觸發**(flip_onset=None,因 mask 是 plateau 非高速跳),所以鉤沒被砍。
- **解法**(使用者提案):**速度低於門檻就停止畫軌跡線**。`auto_reconstruct`:新增 `settle_frame(motion, after, min_speed, sustain=3)`——撞擊後速度連續 `sustain` 幀 < 門檻即視為停住。`build_data` 改為**對每台車**取 `min(settle, flip[僅 stop_vehicle], impact+1 fallback)` 截斷。
- **設定**:`SceneConfig.min_traj_speed_kmh`(override `min_traj_speed`,預設 3.0 km/h,0=關閉);step④ 設定列加「停止畫線速度」輸入。
- **驗證**:警車 140-204→**140-178**(砍掉停後鉤,路徑 17.3→12.4 m)、計程車 120-245→**120-212**(砍掉停妥抖動);圖上兩線乾淨收尾。
- **注意**:此截斷只影響 2D 圖/KML/CSV。**追蹤影片**的警車框在 f204 消失是 **SAM2 跟丟**(警車翻覆後難追,非刻意截斷)——要在影片續框需在 step③ 對警車補 f210/f230 等校正框;影片 trail 目前未做速度門檻(像素端無公制速度)。

### 問題 1(2D 路線「太直」)現況說明

- 量測:計程車公制路徑整段維持 ~37° 方位 = **本來就接近直線**。影片裡的彎主要是**魚眼透視**,投影到地面被正確攤平;對齊是相似變換(旋轉+縮放)**不會拉直**。故「太直」是投影/校正天花板 + 計程車確實沿義二路直行,非對齊在拉直。要更忠實的曲線需更好的相機標定(已擱置的深層解)。
