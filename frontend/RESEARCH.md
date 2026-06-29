# Three.js 還原車禍路口 — 調研整合與技術決策

> 來源：四個 AI（Claude / DeepSeek / Gemini / Manus）的獨立調研（2026-06）整合。
> 目的：用 Web 3D 還原**真實路口**並讓車輛依後端算出的 `reconstruction.json`
> （公尺座標軌跡＋速度）行駛、可拉時間軸回放、標示撞擊點。
> 本檔是**決策紀錄**，實作以此為準；逐路線細節見最後的連結清單。

---

## 1. 四方共識（先看這段）

四份報告獨立得出高度一致的結論：

1. **底圖逼真度天花板**：Gaussian Splatting（實拍）＞ Google Photorealistic 3D Tiles
    （線上實景）＞ 攝影測量網格 ＞ 純資產 PBR ＞ Mapbox 擠出方塊 ＞ 衛星地形。
2. **不用實地拍攝就能還原真實路口**：只有 **Google 3D Tiles** 能僅靠 GPS 取得實景 3D。
    其餘高逼真路線（Splatting、攝影測量）**都必須到現場拍攝**。
3. **應用外殼/架構**：四方一致 **React Three Fiber (R3F) + drei**——資料驅動動畫、
    時間軸 UI、HDRI/陰影/後處理都最好寫。
4. **被淘汰**：Mapbox/Threebox（只有擠出色塊，不夠實景）、CesiumJS（過重、車材質差、
    雙 renderer 同步煩）、three-geo 衛星地形（路口尺度=一張糊掉的貼圖）。
5. **回放引擎**：用**依 `t_sec` 的時間內插**，**不要**把 Theatre.js 當主回放引擎
    （它是設計師手 K 關鍵影格的工具，與資料驅動不合；且維護曾停滯）。Theatre.js
    只在「相機運鏡潤飾」時才考慮。

> 註：DeepSeek/Manus 一度建議用 `CatmullRomCurve3.getPointAt()` 沿弧長等速取點——
> Claude/Gemini 正確指出**那會抹掉 `speed_kmh` 的加減速**（弧長參數=等速）。
> 結論採後者：**位置以 `t_sec` 內插**保留真實速度；要平滑可用 CatmullRom 只做「位置
> 平滑」、但取點仍以時間驅動。

---

## 2. 路線比較（四方整合）

| 路線                                         | 逼真度                          | 難度           | 需實拍      | 成本/授權                                              | 行動效能   | 契合度           |
| -------------------------------------------- | ------------------------------- | -------------- | ----------- | ------------------------------------------------------ | ---------- | ---------------- |
| **1. Google 3D Tiles + `3d-tiles-renderer`** | 高（航拍級；俯/斜佳，貼地略糊） | 中             | ❌ 只要 GPS | Google Enterprise SKU，~1000 免費/月後計費，須顯示來源 | 中         | **最高**         |
| 2. CesiumJS（含 Google tiles）               | 高（同資料源）                  | 高             | ❌          | 同上 / Cesium ion                                      | 中低       | 中               |
| 3. Mapbox + Threebox                         | 低（擠出方塊）                  | 低中           | ❌          | Mapbox 計費，套件綁舊版                                | 高         | 低               |
| **4. Gaussian Splatting**                    | **最高（近照片）**              | 中高           | ✅ 必須     | 觀看開源；擷取免費～$18/月                             | 中（檔大） | 高（加分項）     |
| 5. 攝影測量 → glTF mesh                      | 高（略遜 splat，好打光）        | 中高           | ✅          | 多免費/低價                                            | 中高       | 中高             |
| **6. 純資產 PBR + 後處理**                   | 中（GTA 級，非該路口實景）      | 中（美術工重） | ❌          | 全開源、無 API 費                                      | 高         | 中（當逼真度層） |
| 7. 衛星地形（three-geo）                     | 低（路口=平地）                 | 低             | ❌          | Mapbox token                                           | 高         | 低               |
| **8. R3F + drei（外殼）**                    | N/A（看底圖）                   | 低中           | 看底圖      | 全開源                                                 | 高         | **必選**         |

---

## 3. 採用的技術決策

**主路線：Google Photorealistic 3D Tiles（`3d-tiles-renderer`，NASA-AMMOS）+ R3F 外殼。**

- 唯一能「零實拍、只靠 `origin_latlon`」拿到該路口實景 3D，且原生 three.js、
    車輛＝普通 `Object3D` 直接吃我們的公尺軌跡、ENU 對位順。
- 取捨：貼地特寫會糊；需 Google Map Tiles API key（計費＋顯示來源＋不可離線散布）。

**逼真度加分層（疊在主路線上，四方一致都會用到）：**

- R3F/drei `<Environment>`（HDRI）、方向光即時陰影、`drei` `<ContactShadows>`
    （讓車不像漂浮）、車漆 `MeshPhysicalMaterial` + `clearcoat`、
    `pmndrs/postprocessing` 的 Bloom（車燈）/ GTAO（接觸陰影）/ ACES tone mapping，
    地面濕滑反射用 `MeshReflectorMaterial`（SSR 在 pmndrs/postprocessing 仍 NYI）。

**極致逼真（選配，針對重點路口）：Gaussian Splatting。**

- 能回現場拍 1–2 分鐘 → Luma/Postshot 產 `.spz`/`.splat` → 用 `@sparkjsdev/spark`
    的 `SplatMesh`（當 `THREE.Object3D` 一等公民，好與車 mesh 混排）或 mkkellogg
    `GaussianSplats3D` 疊進同一場景。需手動對位 2–3 個地面對應點，並鋪一層隱形
    receive-shadow 平面讓車有影子。

**一句話策略**：底圖用 Google 3D Tiles（量產、免拍攝）＋重點路口補 Gaussian
Splatting（極致逼真）＋全部用 R3F 外殼與**自訂時間軸回放**。

---

## 4. 兩個關鍵工程點

### 4.1 資料驅動的車輛回放（不是手 K 關鍵影格）

資料是每車每幀 `{ t_sec, x_m, z_m, speed_kmh }`。做法：

1. 全域 `currentTime`（秒），所有車共用 → 自動同步。
2. 每幀對每車的 `track[]` 用 `currentTime` 在前後兩筆之間**線性內插位置**
    （取樣密＝逐視訊幀，通常不需 spline；要更滑可對位置做 Catmull 平滑，但**取點仍依
    時間**，不可用弧長等速取點，否則抹掉加減速）。
3. 車頭朝向：用相鄰點切線 `lookAt(next)`，抖動用 `Quaternion.slerp` 平滑。
4. 真實時間：`currentTime += delta * playbackRate`（0.25× / 1× / 慢動作）。
    速度快慢會自然呈現（相同 Δt 走更遠）。
5. 時間軸：一條 `<input type=range>` 綁 `currentTime`（0 ~ 末幀 `t_sec`）。
6. 撞擊：`currentTime` 跨過 `impact.t_sec`（或某車 `is_impact`）→ 在 `impact` 公尺座標
    閃爍標記、可自動暫停。

### 4.2 本地公尺平面 → 真實世界（ENU 對位）

`x_m`=東、`z_m`=北、原點 `origin_latlon`——這**正是 ENU 切平面**。

- **Three.js 軸向**：x = `x_m`（東），**z = `-z_m`**（北朝 -Z，俯視時北在上），y = 高度。
- **純場景（PoC 階段）**：原點 `(0,0,0)` 當 `origin_latlon`，直接放車/路/撞擊點。
- **接 Google 3D Tiles**：把整包「公尺場景」放進一個 `originGroup`，用 renderer 的
    `Ellipsoid` / `GeoUtils`（R3F 綁定有 `EllipsoidContext` 的 `frame`）算出
    `origin_latlon` 的 ENU frame，設成該 group 的世界變換（含 `rotation.x = -π/2`
    把 tiles 轉成 y-up）。對位對了，軌跡＋道路＋撞擊點會自動疊在實景路口上。
- **接 Splatting**：splat 是訓練時 SfM 定的任意尺度/朝向，需手動求剛體＋尺度變換
    （拿 2–3 個地面對應點）對齊公尺平面。

---

## 5. 分階段實作計畫

- **Phase 0（本次 scaffold）**：Vite + React + TS + R3F + drei。讀 `reconstruction.json`
    → 畫地面占位 + 道路中心線 + 車輛（占位 box / 之後換 glTF）+ 撞擊標記，
    做好**時間軸回放**（時間內插、車頭朝向、播放/暫停/速度）。**先不接 Google Tiles**
    （免 API key 摩擦），但座標與結構預留好。
- **Phase 1**：加逼真度層（HDRI、ContactShadows、車漆 PBR、後處理 Bloom/GTAO）。
- **Phase 2**：接 Google Photorealistic 3D Tiles（`3d-tiles-renderer/r3f`），ENU 對位。
- **Phase 3（選配）**：重點路口補 Gaussian Splatting（Spark），手動對位＋隱形地面陰影。

PoC 驗收：軌跡/道路/撞擊點正確疊合、車沿線跑且車頭對、`speed_kmh` 快慢有反映、
撞擊瞬間有標示、行動端能順跑。

---

## 6. 套件清單（Phase 0–1）

- 核心：`three`、`@react-three/fiber`、`@react-three/drei`
- 後處理：`@react-three/postprocessing`、`postprocessing`
- 狀態：`zustand`（管 `currentTime` / 播放狀態）
- 之後：`3d-tiles-renderer`（Phase 2）、`@sparkjsdev/spark`（Phase 3）
- 工具：`vite`、`typescript`、`@types/three`

---

## 7. 參考連結（去重精選）

**Google 3D Tiles / 3d-tiles-renderer（主路線）**

- GitHub：<https://github.com/NASA-AMMOS/3DTilesRendererJS>
- Live 範例：<https://nasa-ammos.github.io/3DTilesRendererJS/example/bundle/index.html>
- R3F 綁定：<https://github.com/NASA-AMMOS/3DTilesRendererJS/blob/master/src/r3f/README.md>
- Google 文件/計費：<https://developers.google.com/maps/documentation/tile/3d-tiles>
    ／<https://developers.google.com/maps/billing-and-pricing/pricing-categories>

**Gaussian Splatting（加分層）**

- Spark：<https://github.com/sparkjsdev/spark> ／ <https://sparkjs.dev/>
- mkkellogg：<https://github.com/mkkellogg/GaussianSplats3D>
- 擷取工具實務：<https://swyvl.io/blog/how-to-create-gaussian-splats/>

**R3F / 後處理 / 逼真度**

- R3F：<https://r3f.docs.pmnd.rs> ｜ drei：<https://github.com/pmndrs/drei>
- postprocessing：<https://github.com/pmndrs/postprocessing> ｜
    demo：<https://pmndrs.github.io/postprocessing/public/demo/>
- three.js 車漆範例：<https://threejs.org/examples/#webgl_materials_car>

**沿軌跡移動 / 對位**

- 物件沿路徑（含車頭朝向）：<https://waelyasmina.net/articles/how-to-make-an-object-follow-a-path-in-three-js/>
- `CatmullRomCurve3` API：<https://threejs.org/docs/#api/en/extras/curves/CatmullRomCurve3>

**事故重建（成品/方法參考）**

- 業界桌面標竿：PC-Crash、Virtual CRASH（Windows 桌面，看「成品長相/流程」）。
- 影像→量測級 3D + 撞擊速度（2025）：<https://www.mdpi.com/1999-4893/18/11/707>
- 觀察：目前**無現成 web/three.js 開源事故回放引擎**吃我們這種 JSON——這正是本專案要做的。
