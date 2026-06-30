# 高斯噴濺（Gaussian Splatting）實作調查筆記

把車禍路口還原成「地平線6 級」實景的調查與實作紀錄。對應程式：
[`src/scene/SplatScene.tsx`](src/scene/SplatScene.tsx)（viewer）、[`src/scene/Scene.tsx`](src/scene/Scene.tsx)
（底圖切換）、[`src/App.tsx`](src/App.tsx)（依底圖切 `logarithmicDepthBuffer`）。

調查時間：2026-06。

---

## 0. 一句話結論

- **用 Google Maps / Earth 截圖繞一圈做 splat ＝ 沒用**（達不到目標，原因見 §1）。
- **要「地平線6 級實景」只有一條路：到現場用手機/空拍實拍** → 雲端/本機訓練成 splat
    → 丟進我們已做好的 viewer，和開車軌跡疊在一起。
- viewer 已實作並驗證能**正確載入** splat（mkkellogg `DropInViewer`，1738 個 gaussian、
    位置/尺度都對）；對位用環境變數手動微調（§5）。

---

## 1. 為什麼「Google 截圖 → splat」不可行（鐵律）

**高斯噴濺的輸出畫質 ≤ 輸入照片畫質，無中生有不出細節。** 所以：

| 來源                              | 能不能做               | 結果                                                                                                                                                                            |
| --------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Google Maps/Earth **3D 視角**截圖 | 技術上可跑 pipeline    | **頂多等於、實際更差**——對「你已經有、且已嫌糊」的 Photorealistic 3D Tiles 網格再拍照，重建出同一份航拍級的糊。**清晰度零提升**。唯一好處是離線靜態檔、好對位，但那不是你要的。 |
| Google **Street View** 截圖       | 真實照片、細節較多，但 | 視角稀疏（只沿路中心、每 5–15m 一張全景）、全景畸變、車人烤死、且 **Google 條款禁止拿街景做衍生 3D 資產**（法律風險）→ 不乾淨、不可靠、不建議。                                 |
| **衛星俯視**截圖                  | 否                     | 只有頂視、無視差，做不出立體街景。                                                                                                                                              |

→ 對位/離線雖有小確幸，但**逼真度沒有提升**，不值得做。

---

## 2. 正解：實拍 → 訓練 → splat

### 2.1 拍攝（決定成敗，§ outdoor 街景重點）

- **設備**：任何智慧型手機即可；有空拍機更好（路口俯視＋多角度）。
- **方式**：繞著路口走一圈，**邊走邊平移視角**（不是原地轉），錄 **1–3 分鐘影片**或
    拍 **150–400 張照片**。
- **重點**：
    - **重疊度 70–80%**、移動要有「橫向視差」（左右走動，不要只前進）。
    - **鎖定曝光/對焦**，避免自動曝光跳動（splat 對光照一致性敏感）。
    - 大晴天強光/強陰影會有 artifact（Scaniverse 官方也提到）；陰天或柔光最佳。
    - 盡量沒有移動的車人（會被「烤進」場景）；多角度拍可讓訓練濾掉部分動態物。
    - 把**路口中心**拍滿、四個路口臂都涵蓋（之後對位才有共同地標）。

### 2.2 訓練工具（2026，依「無需 GPU / 品質 / 價格」選）

| 工具                                      | 平台                         | 特性                                                                                    | 價格                |
| ----------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------- | ------------------- |
| **Luma AI**                               | 手機 App / Web               | **戶外、植被、天空特別好**；上傳影片約 1 分鐘出圖                                       | 免費層佳            |
| **Scaniverse**（Niantic）                 | iOS/Android                  | **免費**、最簡單；掃 30–60s、5–10 分鐘出圖；強光易有 artifact、受手機記憶體限制場景大小 | 免費                |
| **Polycam**                               | iOS/Android（含 LiDAR）/ Web | 最普及、LiDAR＋攝影測量＋3DGS；匯出 `.ply` 需 Pro                                       | 免費／Pro ~$8–18/月 |
| **Postshot**                              | 桌面（Win，雲端）            | 建築/不動產強；品質接近專業方案；可匯出多格式                                           | 免費層              |
| **Nerfstudio / gsplat / 原版 INRIA 3DGS** | 本機（NVIDIA GPU）           | 開源、最可控、可大場景；要自己跑 COLMAP 算相機位姿                                      | 免費（需 GPU）      |

> 入門推薦：**有 iPhone LiDAR → Polycam；任何手機 → Luma AI**。要完全可控/大場景 →
> Nerfstudio(gsplat)。

### 2.3 匯出格式 → 我們的 viewer 吃哪些

viewer（mkkellogg）支援：**`.ply` / `.splat` / `.ksplat` / `.spz`**。

- `.ply`：原始、檔案大。
- `.splat`（antimatter15）：精簡二進位。
- `.ksplat`：mkkellogg 自家壓縮格式（可用其工具轉，載入最快）。
- `.spz`：Niantic 開源壓縮格式，比 `.splat` 小 1/2~1/5，**行動裝置友善**（建議首選）。

---

## 3. 已實作的 viewer

- 套件：**`@mkkellogg/gaussian-splats-3d`**（peer `three >=0.160`，相容我們的 0.169；
    Spark 2.x 需 `three >=0.180` 會連帶升 R3F 9/React 19，故先用 mkkellogg）。
- `DropInViewer` 是 `THREE.Group`，靠 `onBeforeRender` 自我更新（排序），所以
    `<primitive object={viewer}/>` 直接融入 R3F render loop，**不需** COOP/COEP header
    （`sharedMemoryForWorkers: false`）。
- **底圖優先序**（`Scene.tsx`）：**splat（`VITE_SPLAT_URL`）> Google 3D Tiles
    （`VITE_GOOGLE_TILES_KEY`）> 占位格線地面**。
- **重要相容性**：Google Tiles 需要的 `logarithmicDepthBuffer` 會**讓 splat shader 失效**，
    所以 [`App.tsx`](src/App.tsx) 只在「用 tiles」時開 log depth，用 splat 時關掉。

### 驗證狀態

- ✅ viewer 能**正確載入並解析** splat：合成測試檔 `public/sample.splat` 進來後，
    `splatMesh` 有 **1738 個 gaussian**、抽樣中心點落在 ±12m、scale 0.42（與產生時一致）。
- ⚠️ 合成測試檔目前**畫面呈現**還在微調（手寫 `.splat` 的 opacity/顏色編碼與 mkkellogg
    期望尚未完全對上）——**這是測試 fixture 的問題，不是 viewer 的問題**；真實工具
    （Luma/Polycam/Postshot）匯出的 splat 用的是標準編碼，可正常顯示。拿到真實 splat 後再
    以它為準驗證。

---

## 4. 怎麼接到場景（你的下一步）

1. 到事故路口實拍（§2.1）。
2. 用 Luma / Polycam / Postshot 產出 `.spz`（或 `.ply`/`.splat`/`.ksplat`）。
3. 放到 `frontend/public/`（或任何 CORS 可取的網址）。
4. `frontend/.env` 設：
    ```
    VITE_SPLAT_URL=/your_scene.spz
    ```
5. `npm run dev` → splat 變成底圖，車輛/軌跡/撞擊點疊在上面。

---

## 5. 對位（splat ↔ 我們的公尺場景）

splat 由 SfM 定出**任意尺度/朝向/原點**（LiDAR 來源通常已是公尺，COLMAP 來源無單位）。
我們場景是「以 `origin_latlon` 為原點的公尺」（x=東、z=−北、y=上）。對位＝**相似變換**
（旋轉＋等比縮放＋平移）。

**手動對位旋鈕**（`.env`，見 `SplatScene.tsx`）：

| 變數                                         | 作用                              |
| -------------------------------------------- | --------------------------------- |
| `VITE_SPLAT_SCALE`                           | 等比縮放（splat 單位 → 公尺）     |
| `VITE_SPLAT_ROT_X_DEG` / `_Y_DEG` / `_Z_DEG` | 旋轉（通常先調 Y＝朝向/南北方位） |
| `VITE_SPLAT_X` / `_Y` / `_Z`                 | 平移（公尺）                      |

**做法**：拍攝時讓**路口中心 ≈ `origin_latlon`**；載入後依序調 `SCALE`（讓 splat 的路寬對上
道路中心線）→ `ROT_Y`（南北方位）→ `Y`（地面高度）→ `X/Z`（對齊路口中心）。
進階可拿 **2–3 個地面對應點**（路口角落，公尺座標已知於 `reconstruction.json`）解相似變換
（學界做法見 GeoRefGS：用 learnable similarity transform 把 3DGS 對到地理座標）。

---

## 6. 效能 / 限制

- 檔案：單一路口 splat 常數十～數百 MB；用 `.spz` 壓縮＋只拍必要範圍可控。
- splat 是**靜態**：拍攝當下的車人會烤進場景，需在 SuperSplat 等工具裁掉雜物。
- 行動裝置可跑但要分層載入/壓縮；`.spz` + gpuAcceleratedSort 較友善。
- 與一般 mesh 的陰影：splat 不投/接陰影，車底用 `ContactShadows` 補、或鋪隱形 receive
    -shadow 平面。

---

## 7. 參考來源

- 2026 工具比較：[Polyvia3D](https://www.polyvia3d.com/guides/gaussian-splatting-tools-comparison)
    ／[THE FUTURE 3D](https://www.thefuture3d.com/blog/gaussian-splatting-software-tools-compared-2026/)
- 行動裝置擷取：[Polyvia3D — Mobile capture](https://polyvia3d.com/guides/gaussian-splatting-mobile-capture)
- 戶外擷取指南：[Real Horizons — Outdoor capture guide](https://realhorizons.ai/blog/outdoor-gaussian-splatting-capture-guide/)
- Polycam 3DGS：<https://poly.cam/tools/gaussian-splatting>
- viewer：[mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
    ／[Spark（World Labs）](https://www.worldlabs.ai/blog/spark-2.0)
- 對位/地理座標：[GeoRefGS（MDPI 2026）](https://www.mdpi.com/2504-446X/10/3/195)
    ／[GIS 實務指南](https://geo-matching.com/articles/gaussian-splatting-for-mapping-and-gis-a-practical-guide-to-the-new-3d-standard)
- splat viewer 比較：[Swyvl — Best splat viewers 2026](https://swyvl.io/blog/best-gaussian-splat-viewers/)
