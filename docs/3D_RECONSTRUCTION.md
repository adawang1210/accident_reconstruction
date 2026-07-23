# 3D 場景重建

事故的**車速與位置**一律以 2D homography 管線為準（見 [`README.md`](../README.md)、
[`summary.md`](summary.md)）。3D 只是把那條 2D 軌跡放進一個**看得懂的路口場景**裡。目前有
**兩條並行的線**，各自完成度不同：

| 線 | 做什麼 | 程式 | 位置 | 級別 |
|---|---|---|---|---|
| **A. 深度 Splat 背景** | 從 CCTV 影片重建路口 3D 點雲（純場景、無車） | `depth_backdrop.py` | **master** | 示意級 |
| **B. CAD 路面模型** | 影片自標定量出路面幾何 → 向量化 CAD 路口 | `self_calibration.py` + `RoadCad.tsx` | branch `accident-scene-cad-modeling-5aaa03`（**未併 master**） | 量測級（±3%） |

> ⚠️ 線 B 的實作在**未併入的 branch** 上；在 master 上看不到 `self_calibration.py` /
> `RoadCad.tsx`。要看要 `git checkout claude/accident-scene-cad-modeling-5aaa03`
> 或進該 worktree。

---

## 線 A — 深度 Splat 場景背景（`depth_backdrop.py`，master）

**不到現場**，直接把案發影片本身重建成路口 3D 點雲：

1. **時間中位數空景**：取撞擊前（~frame 140 前）的幀做逐像素中位數 → 移動車流被平均掉，
   得到「空的路口」。
2. **Inpaint**：洗掉一直停著的休旅車、殘影、路面浮水印。
3. **單目公制深度**（Depth-Anything，MoGe-2 為選項）→ 反投影成點雲，後處理抹平梯田紋
   （`--smooth`）、丟深度陡變的拉絲簾幕（`--edge-filter`）、RANSAC 壓平路面（`--flatten`）。

**產物**（`data/<場景>/`）：`BMW_scene_smooth.splat`（前端用，~2 M 點）、`_dense`/`_clean`/`_v2`、
`BMW_scene_clean.ply`（拖進 <https://superspl.at/editor> 可繞看）、`_orbit.mp4`、
左右視圖 PNG。前端以 `THREE.Points` 繪製（非 gaussian 光柵化，原因見 `frontend/SPLAT_NOTES.md` §13）。

**級別**：**示意級、非量測級**。左右 ±15° 內幾何正確，大角度出現遮蔽破洞（點雲邊緣拉扯）。

**重生指令**與所有參數：見場景內 `data/<場景>/README_3D.md`，以及 `frontend/SPLAT_NOTES.md`
（36 節，含 §11 交通監視器自標定、§12 depth_backdrop、§13 為何用 Points、§15/§16 後處理）。

---

## 線 B — CAD 路面模型（`self_calibration.py` + `RoadCad.tsx`，branch `accident-scene-cad-modeling-5aaa03`）

把路口做成**依實測幾何生成的向量化 CAD 模型**，取代示意的 OSM 底圖。

### 1. 影片自標定（`self_calibration.py`）
不靠任何外部底圖或 GCP：
- 路面標線提供**兩族平行線**——沿路（縱向）與跨路（橫向）。兩組**消失點**定出相機
  焦距與朝向（`focal_from_orthogonal_vanishing_points`：`(v_a−c)·(v_b−c) = −f²`）。
- 一個**已知真實長度**（車輛輪距 ~1.565 m）定出相機高度與尺度。
- 全部從影片來。BMW：焦距 ~3691 px、相機高 **3.73 m**、俯角、HFOV 都量得出。

> 為何不用 GCP homography：GCP 若在底圖上點得太擠，擬合殘差很小卻讓所有真實距離被壓縮
> （BMW 的 ~18 m GCP 跨度正是主因，見 [`summary.md`](summary.md) 車速一節）。消失點法用**整個
> 路面的標線**當約束，不吃這個虧。這就是 [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) 說的「深層解」。

### 2. 量出路面幾何（`scene_records/<場景>_road_cad.json`，schema `road_cad/1`）
以自標定的視角量出並記錄（BMW，2026-07-22）：
- **車道**：右路緣 / 分隔島路緣 / 車道寬（3 車道，~2.12 m/道）。
- **縱向標線**：邊線、車道線（含實測 paint 寬與**實際塗漆段**，非名目虛線節律）。
- **穿越帶**：行人＋自行車帶（X 17.4–22.6 m）。
- **地面座標**：X = 沿路（遠離相機為正）、Y = 跨路（行進方向左側為正，Y≈0 為右路緣）。
  建模範圍 **X 12–70 m**（相機看到 ~138 m，但 70 m 外標線糊成一團量不到）。

### 3. 前端渲染（`RoadCad.tsx` + `roadCadParams.ts`）
把 `road_cad.json` 的實測幾何畫成 3D 路口（`Scene.tsx` 接入，取代 OSM 底圖）。

### 4. 只建路面、不編造離地物件（重要且與 anchor 相關）
平面單應性**只對貼地的點有效**。分隔島擋土牆、高架橋墩、人行道欄杆、對向車道、**車身**都
在地平面之上，故**刻意不建**（`not_modelled`）而非亂編。離地 `dh` 的點會沿視線被推遠：
`Y' = Y·h/(h−dh)`，`h = 3.73 m`。

> 這條 `off_plane_projection_caveat` 正好解釋了「撞擊畫面車輛看起來沒被框到」：看到的是
> **車身**，車輛真正的著地點在車道邊界上——與 2D 管線改用**在地輪廓 anchor**（貼地接地點，
> 見 [`summary.md`](summary.md)）是同一個道理。

**驗證圖**（`data/<場景>/cad/`，開發用、gitignored）：`cad_vs_real.png`（CAD 疊實景，斑馬線／
路緣／箭頭／車道線對齊）、`impact_vs_model.png`（雙車軌跡＋車道幾何疊實景）、`bev_*.png`
（鳥瞰）、`mark_*.png`（標線偵測 tophat/edges）。

**級別**：量測級，但**未 geo-reference**（自標定無北向基準），絕對尺度帶 ±3%（來自那一個已知長度）。

---

## 兩條線的關係與**仍開放**的一哩

- 線 A（splat 背景）給「看起來像現場」的點雲；線 B（CAD）給「量得準」的路面幾何。
- **軌跡 ↔ 3D 場景對位還沒接**：`splat_georef.py`（Umeyama 相似變換）是為此準備的，但
  `README_3D.md` 註明「軌跡尚未與 backdrop 對位」，所以有 backdrop 時前端**預設不畫車**。
  這是 3D 這條線的下一個關鍵步驟。

## 相關 commit / 檔案

- 線 A：`feat(3d): 單張深度 → 3D 示意背景 splat`、`--scene-only 純場景重建`（master）。
- 線 B（branch `accident-scene-cad-modeling-5aaa03`）：
  `feat(cad): 影片自標定取代不可信的 GCP homography`、`依實測幾何生成路口 3D 模型`、
  `標線普查補完路面模型延伸到 70 m`、`分隔島路緣 9.8→10.3 m`。
- 對位（未完）：`feat(3d): splat↔scene georeferencing via Umeyama similarity solve`。
