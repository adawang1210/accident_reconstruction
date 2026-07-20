# 專案技術檢察報告 — 修改與優化建議

檢察日期：2026-07-19。範圍：後端套件 `accident_reconstruction/`（8 個模組）、前端
`frontend/`（Vite + React + R3F）、測試、CI、文件。以「資深貢獻者接手前的盡職調查」
標準逐檔閱讀後彙整。

**總評**：這是一個方法論扎實、文件品質罕見地好的專案。三種框合併失敗模式的根因分析
（`docs/summary.md`）、GCP 涵蓋範圍與速度可信度的誠實揭露、以及「保留辨識曲線、只做
相似對齊」的設計決策，都顯示核心 CV 邏輯經過真實資料的錘鍊。主要的債不在演算法，
而在**架構（import 時綁定的全域狀態）**、**重複程式碼**與**測試覆蓋**。以下按嚴重度
分級，每項附檔案位置與具體修法。

---

## 0. 發現總覽

| 級別          | 項目                                                         | 類型      |
| ------------- | ------------------------------------------------------------ | --------- |
| P0 · 立即修   | CI 只監聽 `main`，但預設分支是 `master` → push CI 從未跑過   | CI        |
| P0 · 立即修   | `web_app` 路徑檢查用字串 `startswith`，同名前綴目錄可穿越    | 安全      |
| P1 · 短期     | KML 輸出未做 XML escape（車輛名稱來自使用者輸入）            | 正確性    |
| P1 · 短期     | import 時綁定 SCENE／校正全域狀態 → 被迫用 subprocess 架構   | 架構      |
| P1 · 短期     | 通用 writer (`birdseye`) 依賴永康 legacy 模組                | 架構      |
| P1 · 短期     | `ensure_readable_mp4` 的 `check=True` 可在追蹤跑完後炸掉     | 穩健性    |
| P2 · 中期     | 速度視窗邏輯、地理常數、ffmpeg 搜尋等多處重複                | 重複      |
| P2 · 中期     | `overrides.json` 白名單設計本質脆弱（已踩過雷）              | 設計      |
| P2 · 中期     | 撞擊幀在 3 個地方各自重算、`_vehicle_colors` 迴圈內重讀 JSON | 效能/一致 |
| P2 · 中期     | 測試缺口：核心幾何/偵測函式（`detect_impact` 等）零覆蓋      | 測試      |
| P2 · 中期     | mypy 24 個既有型別錯誤（manual stage，已知債）               | 品質      |
| P3 · 有空再說 | 前端 slerp 依賴 framerate、GET 端點有寫入副作用等            | 品質      |
| 決策          | 兩個前端 worktree 要選定正本；editable install 陷阱的治本    | 流程      |

---

## 1. P0 — 立即修

### 1.1 CI 從未在 push 觸發

[ci.yml](../.github/workflows/ci.yml) 寫的是：

```yaml
on:
  push:
    branches: [main]
```

但本 repo 預設分支是 **`master`**。也就是直接 push 到 master 的 commit（本專案的主要
工作模式，見 git log）**從未跑過 CI**；只有開 PR 才會。修法一行：

```yaml
branches: [master]
```

順帶建議：CI 目前只跑 lint + smoke test，前端完全沒進 CI。至少加一個
`cd frontend && npm ci && npm run build`（`build` 已含 `tsc`，等於免費的型別檢查）。

### 1.2 `web_app` 路徑穿越（同名前綴目錄）

[web_app.py:240](../accident_reconstruction/web_app.py:240)（`/media`）、
[web_app.py:275](../accident_reconstruction/web_app.py:275)（`/api/frame`）、
[web_app.py:297](../accident_reconstruction/web_app.py:297)（`/api/crop`）都用：

```python
if not str(target).startswith(str(DATA_ROOT.resolve())) ...
```

字串前綴檢查擋不住**同前綴的兄弟目錄**：`/media/../data_backup/x.mp4` 解析後是
`…/data_backup/x.mp4`，它 `startswith("…/data")` 為 True，照樣被服務出去。伺服器雖
預設綁 `127.0.0.1`，但這是三處複製貼上的同一顆地雷，且修法是標準庫一行：

```python
if not target.is_relative_to(DATA_ROOT.resolve()) or not target.is_file():
```

（Python ≥3.9，`requires-python >=3.10` 沒問題。）建議抽成一個
`_safe_data_path(relpath) -> Path | None` helper，三個端點共用，順便消掉重複。

---

## 2. P1 — 短期（下一個開發循環內）

### 2.1 import 時綁定的全域狀態 — 本專案最大的架構債

現況的因果鏈：

1. `scene_config.SCENE` 在 **import 時**由 `ACCIDENT_SCENE` 環境變數定死
    （[scene_config.py:448](../accident_reconstruction/scene_config.py:448)）。
2. `calibrate_homography` 在 **import 時**把校正載進模組全域
    `VIEW_TRANSFORMER / ORIGIN_LATLON / USING_GPS_CALIBRATION`
    （[calibrate_homography.py:483](../accident_reconstruction/calibrate_homography.py:483)）。
3. `prompt_track_accident`、`auto_reconstruct`、`birdseye_manual_annotation`、
    `recognized_route` 又各自在 import 時從 `SCENE` 衍生一串模組常數
    （`SOURCE_VIDEO`、`TRUE_IMPACT_LATLON`、`GEO_READY`…）。
4. 結果：`web_app` 要跑「使用者選的場景」只能 **spawn subprocess 重新 import**
    （[web_app.py:495](../accident_reconstruction/web_app.py:495) 的 `_start_job` docstring
    自己也承認這一點）；`run_pipeline` 要覆蓋車輛框只能 **monkeypatch**
    `track.INIT_VEHICLES`（[run_pipeline.py:117](../accident_reconstruction/run_pipeline.py:117)）。

這不是壞掉，但它讓每個新功能都得繞著全域狀態走，測試也難寫（想測不同場景就得改環境
變數重 import）。建議的收斂路徑（可以漸進做，不必大爆改）：

- 新增一個 `Calibration` dataclass（homography、distortion、origin、span），提供
    `Calibration.load(scene) -> Calibration | None`；`ViewTransformer` 掛在它身上。
    `metric_to_latlon` 變成它的 method。
- stage 函式一律吃明確參數：`track.main(scene, init_vehicles)`、
    `reconstruct.main(scene, calibration)`。模組層常數保留為
    `SCENE` 的 default 值即可（CLI 行為不變），但函式內不再讀全域。
- 做完之後 `web_app` 可以改成 in-process 呼叫（丟進 thread/executor），subprocess
    只留給「要隔離 SAM2 記憶體」這一個正當理由。
- 各檔開頭的 `sys.path.insert(0, …)`（`calibrate_homography.py:38`、
    `prompt_track_accident.py:37`、`web_app.py:32`、`birdseye_manual_annotation.py:11`）
    在套件已 editable 安裝的前提下是冗餘的，可全部移除。

### 2.2 通用 writer 依賴永康 legacy 模組

[birdseye_manual_annotation.py:19](../accident_reconstruction/birdseye_manual_annotation.py:19)
從 `manual_pre_impact_motorcycle_annotation`（1115 行的場景專屬 legacy）import 了
`BIRDSEYE_PX_PER_M / MANUAL_TRACKS / create_metric_birdseye_base / metric_to_panel` 等。
這違反了 AGENTS.md 自己立的規矩「場景資料進 `SceneConfig`，stage 模組不寫死場景」：
任何場景跑 `write_kml / write_map_figure / write_csv`（所有場景共用的輸出路徑）都會
**執行永康模組的 import**。建議：

- 把真正共用的（`metric_to_panel`、`create_metric_birdseye_base`、`BIRDSEYE_PX_PER_M`）
    抽到中立模組（如 `panel.py`），legacy 模組反過來 import 它。
- `collect_vehicle_motion / write_birdseye_split_video / write_summary_image` 這幾個
    只有永康在用的函式移回 legacy 模組，`birdseye` 只留通用 writer。
- 另外 `birdseye_manual_annotation.py` 是唯一**沒有模組 docstring** 的模組（第 1 行
    直接 `from __future__`），與專案自訂規範不符。

### 2.3 `ensure_readable_mp4` 可在最貴的計算完成後炸掉

[prompt_track_accident.py:72](../accident_reconstruction/prompt_track_accident.py:72) 用
`subprocess.run(..., check=True)`。SAM2 追蹤跑 3.5 分鐘後，若 ffmpeg 轉檔失敗（磁碟滿、
編碼器缺），整個 `main()` 直接 raise，**CSV 已寫但 exception 蓋掉了成功訊息**（實際上
CSV 是在轉檔之後才寫——所以更糟：追蹤結果全丟）。修法：

- `try/except subprocess.CalledProcessError`，失敗就留原檔並印警告（docstring 本來就說
    「ffmpeg 不在就保留原檔」，失敗理應同樣降級）；
- 或把 `write CSV` 挪到轉檔之前，確保最貴的產物先落地。

### 2.4 KML 未做 XML escape

[birdseye_manual_annotation.py:533](../accident_reconstruction/birdseye_manual_annotation.py:533)
`_kml_linestring` 把 `name` 直接內插進 XML。車輛名稱來自 `vehicle_boxes.json`，是使用者
在工作台輸入的：名字帶 `&`、`<`（例如 `A&B car`）就會產出壞掉的 KML，Google Earth 匯入
失敗。`recognized_route.write_recognized_kml` 同病。修法：`xml.sax.saxutils.escape(name)`
一行。

---

## 3. P2 — 中期（重構與品質）

### 3.1 重複程式碼盤點

| 重複內容                    | 位置                                                                                                                                                                                                                 | 建議                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 速度視窗（0.6 s 位移/時間） | `auto_reconstruct.windowed_motion`（deque 版）與 `birdseye.aligned_motion`（list 版），連 `SPEED_WINDOW_SECONDS = 0.6` 都各宣告一次                                                                                  | 抽成 `motion.py`：一個吃「frame → (x, y) 或 (lat, lon) + 距離函式」的實作                                    |
| 地理常數 `111195.0`         | 全套件 **8 處**（`birdseye` ×4、`recognized_route` ×4），與 `calibrate_homography._METERS_PER_DEG_LAT`（用 radians 算，值同）並存                                                                                    | 統一由 `calibrate_homography` 匯出；順便統一 `_EARTH_RADIUS_M`（6_371_000 vs `_haversine_m` 內的 6371000.0） |
| ffmpeg 搜尋                 | `prompt_track_accident._find_ffmpeg` 與 `web_app._ffmpeg_location`，候選清單相同、一個回檔一個回目錄                                                                                                                 | 合併成一個 util，回 `Path`，呼叫端自取 `.parent`                                                             |
| 撞擊幀計算                  | `auto_reconstruct.build_data`、`recognized_route.build_reconstruction`、`write_recognized_csv`、`prompt_track.main`（truncate 分支）各自 `impact_frame_override or detect_impact(project_metric(load_anchors(...)))` | 抽 `resolve_impact_frame(scene) -> int \| None`，一處實作、一處快取                                          |
| `to_px` 平面投影閉包        | `recognized_route.write_recognized_figure`、`birdseye.write_map_figure` 幾乎相同                                                                                                                                     | 抽一個小型 `MapProjection` helper                                                                            |
| CSV schema 字串             | `birdseye.write_csv` 與 `recognized_route.write_recognized_csv` 同 schema 手寫兩次                                                                                                                                   | 共用一個 writer，吃 `aligned` dict                                                                           |

### 3.2 `overrides.json` 白名單設計

[web_app.py:657](../accident_reconstruction/web_app.py:657) `save_overrides` 的模式是
「**整檔重寫，僅保留硬編碼白名單**」——AGENTS.md §4.5 已記錄這踩過雷（新增 key 忘了
加白名單就被靜默洗掉）。這是設計問題不是實作問題：**預設丟棄、例外保留**的方向注定
持續踩雷。建議反轉為**預設保留、只覆蓋表單有的欄位**：

```python
merged = dict(scene.overrides)  # 現有全部保留
for key in STEP4_FORM_KEYS:  # 這個表單「擁有」的欄位才覆蓋/清除
    value = request.overrides.get(key)
    if value in (None, ""):
        merged.pop(key, None)
    else:
        merged[key] = value
```

`save_anchors` 已經是這個 pattern（只動自己的兩個 key），統一即可。做完後 AGENTS.md
的那條陷阱可以整段刪掉——**用設計消滅陷阱，比文件提醒可靠**。

### 3.3 效能小刀

- **`_display_for` 每次呼叫重讀 JSON**：
    [recognized_route.py:170](../accident_reconstruction/recognized_route.py:170) 在
    per-vehicle 迴圈裡呼叫 `_vehicle_colors()`，而它每次 `read_text + json.loads`
    `vehicle_boxes.json`。呼叫端先讀一次傳進來即可。
- **`discover_scenes()` 在 import 時 `rglob` 整個 `data/`**
    （[scene_config.py:445](../accident_reconstruction/scene_config.py:445)）：`data/` 放了
    大量影片/輸出後，每個 Python 進程（含 web_app 每個 subprocess job）都付一次全樹掃描。
    建議 scene.json 集中放固定淺層（如 `data/*/scene.json` 用 `glob` 而非 `rglob`），或
    延遲到第一次查找。
- **`_segment_masks` 每段 re-encode 一個暫存 mp4**（mp4v 全段重編碼）：目前可接受，
    但若之後段數變多，可考慮改餵 frame 序列或預先切好的共用 clip 快取。

### 3.4 測試缺口

現有 4 個測試檔守住了框合併／predictor 重設／scene records，很好。但**整條幾何與
偵測鏈是零覆蓋**，而它們全是純函式、最好測：

| 未測目標                                      | 為什麼值得測                                                                                 |
| --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `detect_impact`                               | 兩條規則（closest-approach vs first-under-threshold）分支多，docstring 已有現成範例可轉 test |
| `windowed_motion` / `aligned_motion`          | 兩實作行為應一致——正好用 property test 鎖住「重構合併時不變」                                |
| `flip_onset` / `settle_frame`                 | 截斷邏輯直接決定輸出圖，且有 frame-gap 邊界（`gap = max(frame-previous, 1)`）                |
| `_similarity_transform` / `build_alignment`   | SVD 反射分支（`det < 0`）、scale 分支 vs rotation-only 分支                                  |
| `occlusion_corrected_anchor` / `anchor_boxes` | docstring 範例已寫好，一行 `--doctest-modules` 就能收割                                      |
| `load_init_vehicles`                          | 兩種 on-disk 格式的解析                                                                      |
| `build_calibration`                           | span_warning 觸發、MAGSAC fallback 分支                                                      |
| web_app 端點                                  | 用 FastAPI `TestClient` 測 `/media` 路徑檢查（正好回歸 §1.2）、overrides 保留邏輯            |

另外 [pyproject.toml:102](../pyproject.toml) 已設 `doctest_optionflags`，但 `addopts`
沒開 `--doctest-modules`——docstring 裡精心寫的可執行範例目前**一個都沒被跑**。
註：範例用的是 markdown code fence 格式而非 `>>>`，若要收割 doctest 需一併改寫格式，
或改用 pytest 明確測試取代。

### 3.5 mypy 既有 24 個型別錯誤

pre-commit 已設為 manual stage（不擋 commit、CI 不跑）。這是已知債，但「手動才跑」
的檢查等於不存在。建議：修完後改回自動 stage；修不完就 per-module 豁免
（`[[tool.mypy.overrides]] ignore_errors`），至少讓**新程式碼**受檢。核心 dict 型別
（`per_vehicle: dict[str, dict[int, tuple]]` 裡的裸 `tuple`）值得定義
`TrackRecord = tuple[Box, Anchor, Mask]` 型別別名，錯誤會少一半。

### 3.6 其他正確性小項

- **廣 `except Exception` ×5**：最值得改的是
    [auto_reconstruct.py:497](../accident_reconstruction/auto_reconstruct.py:497)——
    recognised 輸出失敗只印一行 `(recognised figure skipped: {error})`，smoke test 的
    docstring 都承認這種靜默曾吞掉 import 錯誤。至少 `traceback.print_exc()`。
- **GET 端點有寫入副作用**：`/api/gcps`、`/api/scene`、`/api/run/status` 經
    `_scene_for_video` → `_register_dynamic_scene` 會 **mkdir + 寫 scene.json**。查詢
    不該落盤；把「建立」拆成顯式 POST，查詢路徑查不到就回 None。
- **`OUTPUT_MARKERS` 名稱片段過濾**（[web_app.py:115](../accident_reconstruction/web_app.py:115)）：
    來源影片檔名含 "tracked"/"route" 等字樣就會從選單消失。改為只排除**已知輸出目錄/
    後綴**（如 `VIDEO_DIR` 下符合 `f"{scene.name}_{suffix}"` 模式者）較穩。
- **`DownloadRequest.folder` 仍接受任意資料夾**，但 UI 已不再詢問——可以移除該欄位
    收窄攻擊/混亂面。
- **`_JOBS` 無鎖**：兩個併發 `/api/run` 可能同時通過 `job["done"]` 檢查、各起一個
    subprocess。單人工具風險低，但一個 `threading.Lock` 三行就能了事。

---

## 4. 前端（`frontend/`）

整體乾淨：型別鏡射後端 schema、時間內插保留加減速、basemap 優先序清楚。發現：

1. **【決策】兩個 frontend worktree 分叉**：HANDOFF §4 已指出本 worktree 與
    `stoic-dijkstra-b59875`（OSM 底圖 + 修好的 tiles）並存。**這是目前前端最大風險**——
    每多一個 commit 就多一分合併成本。建議立刻選定正本、把另一支的差異 cherry-pick
    過來、刪掉舊 worktree。
2. **朝向 slerp 依賴 framerate**：
    [Vehicle.tsx](../frontend/src/scene/Vehicle.tsx) `grp.quaternion.slerp(target, 0.25)`
    每 render frame 固定吃 0.25——120 Hz 螢幕轉頭速度是 60 Hz 的兩倍。改
    `useFrame((_, delta) => …)` 用 `1 - Math.exp(-k * delta)` 當係數即可。
3. **前端零 lint/測試/CI**：`sampleTrack` 的二分搜與 clamp 行為值得幾個 vitest 單元
    測試；`npm run build` 進 CI（見 §1.1）。
4. **Google Maps API key 需加限制**：key 走 `VITE_GOOGLE_TILES_KEY` 打進 bundle 是
    Maps 平台的正常用法，但請確認 key 在 GCP console 上有 **HTTP referrer 限制 + 只開
    Map Tiles API**，否則外流就是別人的免費額度。
5. 小項：`useReconstruction` 沒有 loading/retry 區分（首載 404 時使用者只看到錯誤，
    須手動重整——pipeline 跑完後可加「重試」按鈕）；`store.ts` `let t` 可為 `const`。

---

## 5. 方法論與準確度（非 bug，價值最高的優化方向）

> 車速辨識的完整診斷與優化/替代方案，已擴充為專題章節 **§8**；本節保留一般性項目。

1. **速度不確定度應該進輸出，不只進 stdout**：`print_speed_reliability` 只印在
    terminal，但 KML/CSV/figure 上的速度數字會被單獨截圖引用。建議把
    `gcp_ground_span_m` 與可信度等級寫進 figure 角落與 CSV header 註解——
    `reconstruction.json` 已含此欄位（前端也已顯示警告），2D 輸出應跟上。
2. **GCP span 不足時的軟性補強**：BMW 場景的教訓是「殘差小 ≠ 尺度對」。在
    `build_calibration` 可加第二道檢查：**車輛軌跡的像素範圍 vs GCP 的像素凸包**——
    軌跡跑出凸包多少百分比，直接量化外插程度，比單一 span 門檻更有針對性。
3. **已知幾何當免費 GCP**：路寬（`road_width_m` 已在 SceneConfig）、斑馬線標準寬、
    車道線間距都是現成的尺度約束，可作為校正的 soft constraint 或至少當 sanity check
    自動比對（投影後的路寬 vs 設定路寬，偏差>30% 就警告）。
4. **`detect_impact` 的 `CONTACT_THRESHOLD_M = 3.0` 在壓縮尺度下會失真**：註解已自知
    （「under a compressed homography everything reads close」）。既然 `target_span_m`
    可得，可把 threshold 依尺度可信度調整，或在 span 警告時只信 closest-approach 規則。

---

## 6. 文件與流程

1. **文件重疊**：`README.md`、`docs/README.md`、`docs/PROJECT_SUMMARY.md`、
    `docs/ACCIDENT_2D_RECONSTRUCTION.md`、`docs/summary.md`、`docs/HANDOFF.md` 六份
    互有重疊（pipeline 架構至少寫了三次）。HANDOFF 已是好的入口；建議明確分工：
    HANDOFF（現況+續作）、summary（框合併專題）、其餘合併或標註 archived，過期內容
    刪除比保留好。
2. **editable install 陷阱的治本**：目前靠 AGENTS.md/記憶提醒「worktree 改動要先併
    master 才生效」。兩個治本選項：
    - 在 `run_pipeline` 啟動時印出
        `print(f"running from {Path(accident_reconstruction.__file__).parent}")`，讓
        「跑到的是哪份 code」永遠可見（一行，建議立即做）；
    - 或每個 worktree 自建 venv（`uv sync` 很快），徹底消除共享 venv 的歧義。
3. **`data/` 只存在主 repo** 同理——上面那行 print 也順便印 `cwd`，兩個陷阱一次可視化。

---

## 7. 建議執行順序

| 順位 | 事項                                                    | 工作量  | 依據     |
| ---- | ------------------------------------------------------- | ------- | -------- |
| 1    | CI 分支改 `master`；加前端 build job                    | 10 分鐘 | §1.1     |
| 2    | `_safe_data_path` 修 3 處路徑檢查 + TestClient 回歸測試 | 30 分鐘 | §1.2     |
| 3    | `ensure_readable_mp4` 降級處理；KML escape              | 30 分鐘 | §2.3/2.4 |
| 4    | 決定前端正本 worktree，合併另一支                       | 半天    | §4.1     |
| 5    | `save_overrides` 改「預設保留」；刪 AGENTS 對應陷阱條目 | 1 小時  | §3.2     |
| 6    | 重複程式碼合併（速度視窗、地理常數、ffmpeg、撞擊幀）    | 1 天    | §3.1     |
| 7    | 純函式測試補課（detect_impact 等 8 項）                 | 1–2 天  | §3.4     |
| 8    | 全域狀態收斂為顯式參數（漸進，先 `Calibration` 物件）   | 2–3 天  | §2.1     |
| 9    | birdseye 與 legacy 模組解耦                             | 1 天    | §2.2     |
| 10   | mypy 清 24 錯、改回自動 stage                           | 1 天    | §3.5     |
| 11   | 速度不確定度進輸出、路寬 sanity check                   | 1 天    | §5       |

1–3 是「今天就修」等級；4–5 防止持續性損耗；6–9 是讓下一個功能（真實 splat 對位、
in-process pipeline）站得穩的地基；10–11 提升交付物的可信度——對一個以「事故重建」
為名的專案，輸出數字的誠實度就是產品本身。

---

## 8. 專題 — 車速辨識：診斷、優化與替代方案

（2026-07-19 增補，含網路調研。）現況：路徑形狀正確，但部分場景車速嚴重偏低
（BMW 場景讀到 ~2–8 km/h，實際 ~40–60）。本章先把速度計算鏈逐環拆開、指認每環的
誤差來源，再給出三個層級的修法——從「不改演算法、先修測量基礎」到「換掉 homography
的替代路線」——並附文獻依據與針對本 codebase 的落地步驟。

### 8.1 先搞清楚：速度是怎麼算出來的、每一環的誤差

目前的速度鏈：

```text
SAM2 遮罩底邊中點 (pixel)                     ← 誤差源 C：anchor 抖動/遮蔽
  → homography 投影到公制地面 (east_m, north_m)  ← 誤差源 A：尺度外插（主因）
  → 0.6 s 視窗位移 / 時間                       ← 誤差源 B：時間軸假設 fps 恆定
  → km/h
（geo-ready 場景另有 aligned_motion：從對齊後 lat/lon 以 haversine 重算——已修正）
```

**誤差源 A — 尺度外插（已診斷的主因，`docs/summary.md` 有完整記錄）**：homography
只在 GCP 涵蓋的範圍內可信。GCP 擠在 ~18 m 的小塊地面時，殘差再小（mean 0.33 m）也
只代表「那塊小地毯上是對的」；車輛行經範圍遠超出去，平面投影在校正區外壓縮遠處距離
10–20 倍，速度跟著被吃掉。**重要澄清：這不是演算法 bug，是測量設計問題**——所以
優先順序上「補測量」排在「換演算法」前面。

**誤差源 B — 時間軸（目前 codebase 完全沒防禦，且很可能是第二大誤差）**：
`t_sec = frame / SCENE.fps` 假設每一幀等間隔。但法醫影像分析文獻明確指出：**監視器
DVR 與網路轉載影片（本專案影片全部來自 YouTube 下載）常見變動幀率（VFR）、掉幀、
重複幀**——標稱 30 fps 的檔案，實際幀間隔可以參差不齊；掉 10% 的幀就是速度系統性
高/低估 10%。法醫實務的標準做法是**改用容器內每幀的 PTS 時間戳**（精度可到 μs 級），
而非「幀號 ÷ 標稱 fps」。本專案另有一個自製風險：`trim_clip` 逐幀重寫 mp4 給 SAM2，
若來源有重複幀/VFR，重寫後幀號與原始時間的對應就更不可考。

**誤差源 C — anchor 抖動與遮蔽**：0.6 s 視窗已有效抑制；遮蔽修正
（`occlusion_corrected_anchor`）也已處理。相對 A、B 是小項。

**已修的部分要肯定**：`aligned_motion`（速度改從對齊後路徑以 haversine 重算）已把
**geo-ready 場景**修好（keelung 18/25 → 34/65 km/h）。所以真正的殘餘缺口是：
**沒有道路對齊資料的場景（如 BMW）只能吃原始 homography 尺度**——§8.2 的方案就是
針對這個缺口。

### 8.2 修法（三個層級，依成本/效益排序）

#### Tier 0 — 不改演算法，先把測量基礎修對（本週可做）

**(a) 改用每幀 PTS 時間戳（治誤差源 B；~半天）**

用 ffprobe 抽出每幀的實際時間，取代 `frame / fps`：

```bash
ffprobe -select_streams v:0 -show_entries frame=pts_time -of csv source.mp4
```

落地：`prompt_track_accident` 寫 tracks CSV 時加一欄 `t_sec`（查 PTS 表）；
`windowed_motion` / `aligned_motion` 的時間差改用 `t_sec` 而非 `(frame差)/fps`；
`reconstruction.json` 的 `t_sec` 同步受惠。順便印出「實際幀間隔的變異係數」——
超過幾 % 就警告使用者這支影片時間軸不可靠。這是法醫級影片測速的標準前置步驟
（Epstein & Westlake, *J. Forensic Sciences* 2019）。

**(b) 法定標線 = 免費的高密度 GCP（治誤差源 A 的根因；最高槓桿）**

BMW 場景卡住的根本原因是「衛星圖上讀不到足夠分散的可靠點」。但**尺度校正其實不需要
絕對經緯度，只需要沿路的真實距離**——而台灣道路上到處都是法定尺寸的標線：

| 標線                 | 法定尺寸                            | 依據                              |
| -------------------- | ----------------------------------- | --------------------------------- |
| 車道線（白虛線）     | 線段 4 m、間距 6 m（一週期 = 10 m） | 道路交通標誌標線號誌設置規則 §182 |
| 行車分向線（黃虛線） | 線段 4 m、間距 6 m                  | 同規則 §165                       |
| 路面邊線／行人穿越道 | 線寬等亦有法定值，可現場核對        | 同規則                            |

沿車輛行經方向每條虛線點兩個端點，GCP 的**真實涵蓋範圍立刻拉到整段路**，正中
`docs/summary.md` 開出的藥方（「GCP 要散佈在車輛行經的整段路」），而且不用再到
Google Maps 一個個讀座標。落地方式二選一：

- **簡單版（建議先做）**：工作台加「沿路距離校正」模式——使用者沿路點一串虛線
    端點並標注「這兩點間 4 m／這兩點間 10 m」；後端把這些點以**首點為原點、沿路方向
    為軸**合成局部公制座標，與既有衛星 GCP 一起餵進 `build_calibration`（絕對定位靠
    衛星點、尺度靠標線點）。
- **進階版**：`build_calibration` 支援「pairwise 距離約束」殘差項（兩像素點投影後
    的距離 vs 標定距離），用非線性最小平方（`scipy.optimize.least_squares`）同時解
    homography + k1。

**(c) 尺度覆蓋率檢查與速度不確定度（已在 §5.1/5.2 提出，這裡歸隊）**：軌跡像素
凸包 vs GCP 凸包的覆蓋率、速度 ± 區間進 figure/CSV/JSON。讓「不可信的速度」在
每個輸出物上都自我聲明。

#### Tier 1 — 強化現有幾何路線（1–2 週）

**(d) 已知車輛尺寸當「移動的尺標」（對 BMW 這類無 GCP 場景特別有效）**

事故車的車型通常已知（警方紀錄/影片可辨識），軸距、車長是公開規格。兩種用法：

- **輕量版（半天，先做這個）**：純 sanity check——在車輛行經的幾個位置，比較
    「車輛像素長度經 homography 投影出的公尺長」vs「該車型真實車長」。比值就是該
    位置的**局部尺度誤差係數**，直接印出來（BMW 場景預期會看到 10–20×）。這把
    「速度不準」從猜測變成量化證據，也可反過來當粗略的速度修正係數。
- **完整版**：法醫文獻的 wheelbase/cross-ratio 法——分割出前後輪接地點，用已知
    軸距與投影幾何逐幀解尺度（*Multimedia Tools and Applications* 2026 的遮蔽場景
    法醫測速框架即此路線）。SAM2 遮罩已經有了，加一個輪心估計即可。

**(e) 消失點自動校正（學界對固定交通攝影機的主流解）**

BrnoCompSpeed 這條研究線（Dubská → Sochor → Transform3D → 2025 efficient 方法）
證明：從車流軌跡與車身邊緣自動偵測**兩個消失點**即可解出焦距與相機姿態，再用一個
已知長度（車道線、平均車型尺寸）定尺度——在 21 小時真實測速資料集上，中位數速度
誤差從 7.87 km/h（早期）一路做到 **0.58 km/h**（2025），3D 模型對位定尺度版本
1.10 km/h。我們的場景正是固定 CCTV，完全適用。價值：**擺脫對衛星圖讀點的依賴**，
每支新影片自動出尺度。成本：要實作消失點偵測 + 3D bbox，屬中型功能；建議等
Tier 0 驗證完仍不夠準時再上。

**(f) 分段/加權 homography**：若 GCP 補點後仍集中在兩三塊，可沿路方向拆成多個
局部 homography（或對遠處加權），避免單一全域投影為近處過擬合。比 (e) 便宜，但
治標；有 (b) 之後多半不需要。

#### Tier 2 — 替代/輔助路線（選擇性）

**(g) 反向投影攝影測量（reverse projection photogrammetry）**：法醫界的黃金標準
（Epstein & Westlake 2019；及後續變因研究）——到現場實測參考物、把場景幾何以同型
相機「反向投影」回畫面逐幀對位。最有法庭公信力，但每案人力成本高。**與既有計畫有
綜效**：HANDOFF 待辦本來就要去現場拍 splat——**同一趟現場，順便（1）皮尺實測斑馬
線/虛線距離供 (b) 用、（2）無人機正射圖當高精度底圖取代衛星讀點（讀點誤差可從
~0.5–1 m 降到 \<5 cm）**。強烈建議把這兩件事寫進現場拍攝 checklist。

**(h) 單目 metric depth 模型（Depth Anything V2 / Metric3D / UniDepth）**：
**不建議當主力**——絕對尺度在陌生廣角監視器畫面上誤差常達 5–15%，且無法出具可
解釋的誤差論證；但可當免費的第二意見（投影距離 vs 深度模型距離的一致性檢查）。

**(i) 端到端神經網路直接回歸速度**：需要大量同分布標註、跨場景泛化差、不可解釋
——對以單案重建為目的的本專案不適合，僅記錄為已評估排除。

### 8.3 建議落地順序（車速專線）

| 順位 | 事項                                                      | 工作量 | 預期效果                                 |
| ---- | --------------------------------------------------------- | ------ | ---------------------------------------- |
| 1    | PTS 時間軸 + 幀間隔變異警告（(a)）                        | 半天   | 消除 VFR/掉幀的系統性偏差；所有場景受惠  |
| 2    | 車長尺度 sanity check（(d) 輕量版）                       | 半天   | 把 BMW 的尺度誤差量化成數字、可粗修正    |
| 3    | 工作台「沿路標線距離校正」模式（(b) 簡單版）              | 2–3 天 | 治本：無需衛星讀點即可把尺度拉直到整段路 |
| 4    | 速度不確定度進所有輸出（(c)）                             | 1 天   | 不準的數字不再默默流出                   |
| 5    | BMW 場景以 1–3 重跑驗證（目標：車速落在 40–60 區間）      | 半天   | 用最壞的場景當 acceptance test           |
| 6    | wheelbase 完整版 或 消失點自動校正（(d) 完整版／(e)）     | 1–2 週 | 僅在 1–5 之後仍不達標時投資              |
| 7    | 現場實測 checklist（併入 splat 拍攝行程，(g) 的綜效部分） | 順路   | 一次現場，同時餵 3D 底圖與尺度校正       |

### 8.4 參考來源

- [BrnoCompSpeed：交通攝影機校正回顧 + 測速基準資料集（Sochor et al.）](https://www.researchgate.net/publication/313879450_BrnoCompSpeed_Review_of_Traffic_Camera_Calibration_and_Comprehensive_Dataset_for_Monocular_Speed_Measurement)
- [3D 模型 bounding box 對位定尺度，速度誤差 1.10 km/h（Sochor et al., CVIU 2017）](https://arxiv.org/pdf/1702.06451)
- [Efficient Vision-based Vehicle Speed Estimation（2025，消失點校正+3D bbox，中位誤差 0.58 km/h）](https://arxiv.org/html/2505.01203v1)
- [Determination of Vehicle Speed from Recorded Video Using Reverse Projection Photogrammetry and File Metadata（Epstein & Westlake, J. Forensic Sci. 2019——PTS 時間戳 + 反向投影）](https://onlinelibrary.wiley.com/doi/10.1111/1556-4029.14053)
- [Robust video-based vehicle speed estimation for occluded scenes for forensic analysis（2026——輪心 + 已知軸距 cross-ratio）](https://link.springer.com/article/10.1007/s11042-026-21222-9)
- [Assessing the influence of variables on vehicle speed determination through reverse projection analysis（Science & Justice 2025——幀率不穩是主要時間誤差源）](https://www.sciencedirect.com/science/article/abs/pii/S1355030625001534)
- [道路交通標誌標線號誌設置規則（全國法規資料庫）——§182 車道線：白虛線、線段 4 m、間距 6 m、線寬 10 cm](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=K0040014)
