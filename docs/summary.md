# 碰撞後車輛框合併問題 — 處理摘要

把車禍影片重建成 2D 軌跡時，碰撞後原本分開的兩台車（例如機車＋汽車）會在追蹤
overlay 上「合成一個框」。實際追根究柢，這其實是**三種不同的失敗模式**疊在一起，
分別有不同成因與修法。本檔記錄問題、診斷與最終解法。

相關程式：`accident_reconstruction/prompt_track_accident.py`（Stage 1 追蹤＋渲染）。

---

## 背景

每台車由**獨立的 SAM2 video-memory** 追蹤（各自從使用者框 re-seed），理論上彼此不
共用分類器。所以「合成一個框」不是分類錯誤，而是**遮罩（mask）在碰撞重疊時互相污染**
或**物體在像素上融成一塊**。

---

## 失敗模式與修法

### 模式一：小框塌進大框（被撞車的遮罩外溢到大車）

- **現象**：較小（被撞）車的 SAM2 遮罩外溢、吸附到較大車上，小框長大到與大框重合。
    例如騎士（`person`）被輾後落在汽車框內。
- **成因**：原有的 size 閘門只比相鄰兩幀，逐步外溢每幀 < 1.7× 就溜過去。
- **修法**：`box_containment()` + `merge_suppression_cuts()` — 當小框連續 ≥3 幀
    落入大框 ≥0.6（`MERGE_CONTAINMENT_RATIO` / `MERGE_SUSTAIN_FRAMES`），把小框從合併
    處起丟掉，兩框不再塌成一個。
- **驗證**：`yilan_wujie` 真實資料，`person` 框移除、`car`/`motorcycle` 不受影響。

### 模式二：大框膨脹吞掉融成一塊的另一台車

- **現象**：碰撞後兩車在像素上融成同一連通塊（機車被輾在車頭），存活車的 track 把整塊
    切成一個物件，大框往對方膨脹、蓋住對方。被撞車的 track 在撞擊時就跟丟。
- **成因**：物理融合，單一分割無法把已融成一塊的兩物體再分開。
- **修法（兩種，可並用）**：
    - **`truncate_boxes_at_impact`**（scene override，預設關閉）：撞擊後停止畫/記錄每台車
        的框，只留接近段與撞擊點。適合不想手動標記的場景。
    - **手動補框 re-seed**：使用者在撞擊後幾個關鍵幀手動畫各車的框，SAM2 從那些框重新
        追蹤（見下方模式三的兩個配套修正）。

### 模式三：手動補的撞擊後框失效

使用者照流程在撞擊後手動補了機車框，但輸出仍合併。根因有三：

0. **SAM2 predictor 重用洩漏記憶（真正主因）**

    - `track_vehicle` 原本對同一台車**重用一個 predictor** 跑所有 re-seed 段。這個
        ultralytics 版本（8.4.78）的 `on_predict_start` 並未完全重設 memory bank，導致**後面
        的段被前面段的記憶拉走**：先跑 f97 進場框（左下角）再跑 f160 手動框，f160 竟回傳
        左下角 `(0,626,102,695)` 而非 prompt 的 720–905。所以手動撞擊後框被「拉回進場位置」。
    - **修法**：`_segment_masks` 在每段開頭**重設 predictor 的 `inference_state`／
        `prompts`**，清掉前一段的記憶，各 re-seed 真正獨立。（重用同一 predictor、不重建
        模型——見下方效能優化。）

1. **no-backtrack 閘門誤砍手動框**

    - strict 模式有道「車不會往起點方向倒退」的閘門。被撞機車撞後被輾在車頭，離起點
        的距離（≈939px）比它接近過程的最遠點（≈1144px）還小，被判成「倒退回起點」→
        撞擊後每一幀都丟掉，機車 track 死在撞擊幀。
    - 而且手動框原本只豁免 size 閘門、**不豁免 backtrack 閘門**。
    - **修法**：手動（user-anchored）幀現在**也豁免 backtrack 閘門**，並**重新定義最遠
        距離基準**，使手動 re-seed 整段（含 propagate 出來的幀）都能存活。

2. **大車 track 仍把整塊切成一個，框蓋住機車**

    - 即使機車框救回來，汽車 track 的框/遮罩仍涵蓋機車所在區域。
    - **修法**：`split_overlapping_masks()` — 兩車框重疊時，沿「兩者中心分離最大的軸」
        把較大車的遮罩切掉小車那一側的整條 strip，重算較大車的框，使它**停在**小車邊界
        而非蓋住小車。純幾何後處理、不需 SAM2、不重疊時為 no-op。

- **驗證**：BMW 場景（基隆路四段，BMW 撞機車）真實資料，SAM2 能從手動 f160 機車框切出
    乾淨機車緊框；修正後機車框存活、汽車框停在機車右側，兩框分開。

---

## 設定（per-scene override，`overrides.json`）

| key                        | 作用                                       | 預設   |
| -------------------------- | ------------------------------------------ | ------ |
| `gates`                    | `strict`(預設)/`loose`/`off`，追蹤閘門鬆緊 | strict |
| `truncate_boxes_at_impact` | 撞擊後不再畫框（模式二的自動解）           | false  |

> 注意：`overrides.json` 由 web 工作台的存檔重寫，只保留白名單 key。`save_overrides`
> 的保留白名單已加入 `truncate_boxes_at_impact`，CLI/手動設的值不會被 UI 洗掉。

---

## 手動補框 re-seed 流程（撞擊後要兩個分開的框時）

1. 在工作台把時間軸拉到撞擊後幾個關鍵幀（例如每 5–10 幀一個）。
2. 為每台車各畫一個框（機車框在被輾位置即可，與汽車重疊沒關係）。
3. 存檔（寫進 `vehicle_boxes.json` 的 `objects[].boxes`，支援同物件多幀多框）。
4. 重跑 → 每台車從手動框重新 SAM2 追蹤；手動框不會被 merge 閘門或 backtrack 閘門砍掉，
    且大框會被 `split_overlapping_masks` 裁到不蓋住小框。

---

## 車速判定（已知限制，非 bug）

**症狀**：路徑形狀正確，但車速嚴重偏低（BMW 車讀到 ~2–8 km/h，實際 ~40–60）。

**根因**：車速 = homography metric 距離 / 時間。metric 只在 GCP 校正範圍內可信。診斷
BMW：校正殘差極小（mean 0.33 m）、GCP 兩兩「metric/真實」比值 0.99（區內完美），但
**全部 GCP 只涵蓋 ~18 m 的真實地面**（`target_span_m`）。車輛行經範圍遠大於此，平面
homography 在校正區外（尤其近地平線）會壓縮遠處距離，使 metric 位移被低估 ~10–20 倍
→ 車速跟著被低估。`calibrate_homography.py` 的註解其實已點出此風險。

各場景比較（已校正者尚可，BMW 明顯壞）：

| 場景                  | 車速峰值     | GCP 真實跨度 | 判斷         |
| --------------------- | ------------ | ------------ | ------------ |
| pre_impact_motorcycle | car ~53 km/h | —            | 合理         |
| keelung_xinwu_yier    | ~18–25 km/h  | —            | 偏低但可接受 |
| BMW 神之鬼切          | car ~8 km/h  | ~18 m        | 明顯被壓縮   |

**修法（使用者端，無程式可代勞）**：重新校正，GCP 要**散佈在車輛行經的整段路**（含車輛
進場的遠端、路口兩側），不要擠在一小塊地面。範圍夠大後車速即正確。

**已加的程式守門**：`auto_reconstruct` 現在會印出各車速度峰值與 GCP 真實涵蓋範圍＋
可信度提醒，讓「被壓縮的低速」不再被默默當成真值。

**次要 bug（已修）**：對齊（`_aligned_latlon`）對**位置**套了 per-vehicle scale／道路約束，
但**速度**原本仍用未經 scale 的 homography metric（兩者不一致）。已新增
`aligned_motion()`：直接從對齊後的 lat/lon 以 haversine＋同樣的時間窗重算速度，`write_csv`／
`write_map_figure` 改用它。結果（已驗證）：

| 場景                  | 修正前車速峰值 | 修正後             |
| --------------------- | -------------- | ------------------ |
| keelung taxi / police | 18 / 25        | **34 / 65** km/h   |
| pre_impact（scale=1） | 53             | 53（不變，無回歸） |

注意：此修正只影響 geo-ready（有道路對齊）的 `route_auto` 輸出。BMW 無 geo／無
`true_vehicle_starts`，顯示的是 `route_recognized`（原始投影），速度仍受 homography 尺度
壓縮——**唯有重新校正（GCP 涵蓋整段路）才能修**。

---

## 工作流程陷阱（重要）

- `.venv` 以 PEP 660 editable 安裝把 `accident_reconstruction` 指向**主 repo 的
    master**。worktree 的程式改動**必須先併進 master** 才會在 `python -m ...` 執行；
    `PYTHONPATH` 不會覆蓋 editable finder。
- `data/` 是 gitignored、只存在於主 repo；跑 pipeline 要在主 repo 目錄。

---

## 軌跡 anchor 與平滑化（`ground_footprint.py`）

- **在地輪廓 anchor**（尺度無關）：舊 anchor 是外框底邊中點 `((x1+x2)//2, y2)`，
    那個像素常不在車上（車尾視角浮在路面、轉彎時橫滑）。改取輪廓自己的**中位
    column ＋ 該處接地列**（`contour_anchor_px`），anchor 恆貼在車體接地線上。
    純像素幾何、不假設車輛尺寸，故在尺度不忠實的 BMW homography 上仍有效
    （影像面 anchor 誤差 20→3 px）。
- **已知車長矩形擬合**（尺度相依，選用第二層）：只在 homography 尺度忠實
    （投影輪廓長 ≈ 真實車長的 0.7–1.6 倍）且為箱型四輪車時，再精修到佔地中心。
- **平滑（Kalman+RTS，取代 Savitzky-Golay）**：SG 是局部多項式降噪、**無物理模型**，
    實測後仍讓 BMW **九成的幀不合運動學**（|jerk|>15 m/s³）。改用**等加速 Kalman + RTS
    平滑器**（`trajectory_smoothing.kalman_rts_smooth`）——以連續速度/加速度為先驗，把
    `frac>15` 從 92% 降到 5%（機車 0%）、jerk 降約 60–150 倍，路徑僅移數公分、過彎不切角。
    方法研究、實測數字與踩到的坑（速度初始化、用 innovation gating 而非加速度門檻剔除離群）
    見 [`TRAJECTORY_SMOOTHING.md`](TRAJECTORY_SMOOTHING.md)。
- **形狀平滑（`fit_smooth_curve`，Kalman 之後的最後一段）**：jerk 合格**不代表**圖上是平滑
    曲線。當每幀步長小於 anchor 雜訊（BMW 的 car：192 幀走 5.5 m）時，位置誤差與 jerk 都很小，
    航向卻逐幀甩幾十度——那就是鋸齒。解二階懲罰最小平方（Whittaker 平滑樣條），λ 由軌跡自己
    掃到**每點轉向角 p99 ≤ 5°**且**側向加速度 ≤ 4 m/s²**為止。
- **以撞擊幀分段**：碰撞是脈衝，一條 C² 曲線橫跨撞擊只能靠否認撞擊來達標——實測會把整條軌跡
    拉直去買那個局部特徵（BMW 機車起點搬走 1.77 m、路徑縮 41%）。分段後轉角自然保留。
    另加**形狀保真上限**（任一點位移 ≤ 該段路徑長 15%，量的是相對這一段的輸入）：中位偏離
    對「一端被搬走」不敏感，那次災難的中位數只有 0.22 m。
- **圖上的鋸齒有一部分不在資料裡**：`write_recognized_figure` 原本每幀畫一個半徑 3 px 實心圓，
    車一慢下來就重疊成邊緣不規則的粗塊。改為 `_spaced()`（標記至少間隔一個直徑）。
- **無回歸保護**（per-vehicle）：在地輪廓 anchor 對**寬物體**（汽車）有幫助，但對
    **窄物體**（機車）的中位 column 會逐幀跳、在斷幀處製造假速度尖峰。若某車套用後
    的**峰值速度**超過 legacy 的 1.5 倍就整條退回 legacy（`_PEAK_SPEED_TOLERANCE`）。
    BMW 實測：汽車套用（6→7.9 km/h）、機車退回（避免 54 km/h 假尖峰）。
- **空缺補值 + 虛線橋接**（`interpolate_straight_gaps`）：SAM2 會在車轉向／模糊時
    跟丟數十幀，軌跡與影片線因此消失。\*\*只在空缺前後方向一致（\<25°，直行）\*\*時線性
    內插補回；**跨越轉彎的空缺不補**（直線會切西瓜），保留成虛線。上限 `GAP_FILL_MAX_FRAMES`。
    BMW：直行空缺 41–59／161–179 補上實線（汽車 154→192 點）、轉彎空缺 101–119 留虛線橋接。

## Stage-2 疊加影片（`reconstruction_overlay`，`auto_reconstruct.py`）

Stage-1 的「追蹤疊加影片」（`prompt_tracked`）畫的是**舊的框角 anchor**；Stage-2 另出一支
**重建疊加影片**：把上面校正＋平滑＋補值後的公制軌跡，經 `ViewTransformer.inverse_transform_points`
（反單應＋還原鏡頭畸變）**反投影回原影片畫面**，逐幀畫成長軌跡線＋anchor 點——所以影片上的線
與地圖／CSV 完全一致。工作台「追蹤影片」分頁**優先顯示這支**（`web_app._result_files` 的 `tracked`
kind，沒有才退回 Stage-1 影片）；`auto_reconstruct.main` 會自動產生。線每幀都畫（持續顯示）、
跨空缺處畫**虛線**標明未觀測。

> 3D 場景重建（splat 背景 + CAD 路面模型）另見 [`3D_RECONSTRUCTION.md`](3D_RECONSTRUCTION.md)。

---

## 相關 commit

- `merge_suppression_cuts` / `box_containment`（模式一）
- `truncate_boxes_at_impact`（模式二自動解）＋ web_app 保留白名單
- merge 閘門 anchor-aware（手動框不被砍）
- backtrack 閘門豁免手動框 ＋ `split_overlapping_masks`（模式三）
- 每段重設 predictor `inference_state`/`prompts`（修記憶洩漏，模式三真正主因）
- 效能：重用 predictor＋段長只跑到下一個 re-seed（9 分 → ~3.5 分）
- `aligned_motion`：車速改從對齊後 lat/lon 重算（修速度未套 scale）
- 速度可信度提醒 ＋ AGENTS 陷阱文件 ＋ predictor 重設回歸測試
