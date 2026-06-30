# 高斯噴濺 — 原始研究來源附錄

> 這是**原始資料存檔**：三個 AI（Manus / Gemini / DeepSeek）的完整調查報告，加上我網搜
> 的來源。精煉後的**決策與實作結論**見 [`SPLAT_NOTES.md`](SPLAT_NOTES.md)（§8 為三方整合）。
> 收錄日期：2026-06-30。

---

## A. 我的網搜重點（2026 工具現況）

**手機擷取**

- **Luma AI**：任何智慧型手機上傳影片，~1 分鐘出圖；**戶外、植被、天空特別好**（自家訓練
    pipeline）。
- **Scaniverse（Niantic）**：免費 iOS/Android，「Gaussian Splat」模式掃 30–60s、5–10 分鐘出圖；
    **強光下戶外易有 artifact**、場景大小受手機記憶體限制。
- **Polycam**：最普及（iOS 逾 54 萬評價 4.7★），LiDAR ＋ 攝影測量 ＋ 3DGS。
- 入門推薦：**有 iPhone LiDAR → Polycam；任何手機 → Luma AI**。

**桌面**

- **Postshot**：桌面 + 雲端處理，建築/不動產強，品質接近專業方案。

**對位 / 地理座標化**

- **GeoRefGS**：把 georeferencing 當訓練時的內在約束，用 learnable similarity transform 將
    3DGS 對到全球地理座標系。
- 一般做法：用點雲中的 **ground control points** 與其對應座標，求旋轉＋縮放＋平移變換。

來源連結：

- [Polyvia3D — 工具比較 2026](https://www.polyvia3d.com/guides/gaussian-splatting-tools-comparison)
- [THE FUTURE 3D — 工具比較 2026](https://www.thefuture3d.com/blog/gaussian-splatting-software-tools-compared-2026/)
- [Polyvia3D — 行動擷取](https://polyvia3d.com/guides/gaussian-splatting-mobile-capture)
- [Real Horizons — 戶外擷取指南](https://realhorizons.ai/blog/outdoor-gaussian-splatting-capture-guide/)
- [GeoRefGS（MDPI 2026）](https://www.mdpi.com/2504-446X/10/3/195)
- [GIS 實務指南](https://geo-matching.com/articles/gaussian-splatting-for-mapping-and-gis-a-practical-guide-to-the-new-3d-standard)
- [Swyvl — 最佳 splat viewers 2026](https://swyvl.io/blog/best-gaussian-splat-viewers/)

---

## B. Manus 報告

### B.1 核心：能不能不到現場？

| 來源                  | 可行性 | 畫質 vs Google 3D Tiles    | 法律/授權             | 實例/做法                           |
| --------------------- | ------ | -------------------------- | --------------------- | ----------------------------------- |
| Google Earth Studio   | 極高   | **大幅超越**（Manus 觀點） | 高（ToS 禁商業衍生）  | 匯出 4K 影格 + 相機軌跡(.json) 訓練 |
| Google Street View    | 中     | 劣（近景拉伸、畸變）       | 極高（禁抓取/衍生）   | 補人行道死角，不適合主體            |
| Mapillary / KartaView | 高     | 中（依器材）               | 低（CC-BY-SA 較寬鬆） | API 抓群眾街景序列                  |
| CCTV / 行車紀錄器     | 極低   | 極劣（缺視差）             | 低                    | 單視角無法 3DGS，僅動態疊加參考     |

- 關鍵主張：Google 3D Tiles 是壓縮過的「死網格」；用 Earth Studio 繞行匯出高解析影格重訓
    3DGS，可還原 mesh 丟失的輻射場資訊（玻璃反射、電線）。引 **GBM（Gaussian Building Mesh）**
    論文：Earth Studio 影像 + TSDF Fusion 可產高品質建築模型。
    > ⚠️ 本專案裁決：此「大幅超越」說法**不採**——GBM 是建築 mesh 擷取，非地面近景 photoreal；
    > 詳見 SPLAT_NOTES §8.2。

### B.2 訓練工具

| 工具                | 平台    | 戶外品質 | 價格       | 匯出          | 備註                       |
| ------------------- | ------- | -------- | ---------- | ------------- | -------------------------- |
| Postshot (Jawset)   | Windows | 專業首選 | 訂閱       | .ply/.spz     | 清 artifact 能力最強       |
| Luma AI             | Web/App | 優       | 免費/點數  | .ply/.ksplat  | 自動清背景好               |
| DJI Terra V5.0+     | Windows | 空拍最強 | ~$3,000/yr | .ply/3D Tiles | **直接輸出地理座標**       |
| Nerfstudio (gsplat) | Linux   | 研究級   | 開源       | .ply          | 可控性最高，支援 4DGS 實驗 |

### B.3 對位（7 參數相似變換）

1. 路口找 ≥3 個特徵點（斑馬線頂角、井蓋中心），取 GPS → 轉本地公尺(ENU)。
2. 用 **SplatTransform CLI** 或 SuperSplat 手動縮放對位；或 **GeoRefGS** 訓練時引入可學習
    similarity matrix。
3. three.js 套用：`viewer.addSplatScene(url, { position, rotation(quaternion), scale })`。

### B.4 擷取/編輯最佳實踐

- 避免烤入移動物：離峰時段或無人機高空長時間停留；Postshot 2026 內建「Dynamic Object
    Removal」。
- 對位特徵：拍到路燈桿、交通箱等靜態地標。
- 格式：web 首選 **.spz**（比 .ply 小 ~90%、支援分層流式載入）。

### B.5 鑑識/法律

- 法庭採信取決科學基礎（Daubert Standard）；3DGS 視為「視覺化工具」非「測量數據」。要鑑識
    效力建議搭 **LiDAR 點雲**當幾何真值，3DGS 只當貼圖層。
- ToS：Google 禁止用其影像生成競爭性 3D 數據；學術/私人風險較低，商業需用 Mapillary 等。

### B.6 推薦流程

- **方案 A（不到現場）**：Google Earth Studio 4K Orbit 影格 → Luma/Postshot → SuperSplat 裁切 →
    3 點對位 → R3F + mkkellogg。
- **方案 B（高品質實拍）**：無人機(DJI Mavic 3E)繞拍 + 手機補拍路面 → DJI Terra(帶座標 .ply) →
    轉 .spz → 套 DJI 輸出的地理矩陣。

### B.7 連結

Postshot、SuperSplat Editor、GaussianSplats3D GitHub、Mapillary API、GBM Pipeline Paper。

---

## C. Gemini 報告

### C.1 核心結論：**不行**，要 Forza 級必須實拍（或請代拍）

| 影像來源                  | 可行性  | 優於 Google 3D Tiles？                                | ToS 風險                     | 結論                               |
| ------------------------- | ------- | ----------------------------------------------------- | ---------------------------- | ---------------------------------- |
| Google Earth/Maps 3D 截圖 | ❌ 極低 | **否**（貼地近景已融化，重建只是複製「融化的 3DGS」） | 🛑 極高（禁訓練 ML/衍生 3D） | 原理驗證：3DGS 需多視角高解析影像  |
| Google Street View        | ❌ 低   | 微幅提升但充滿破綻（間距大、缺 70–80% 重疊）          | 🛑 極高（禁抓取/重建）       | 動態物干擾、視角斷層               |
| Mapillary 開源街景        | ⚠️ 中低 | 同上、視角單一（多半只向前）                          | 🟢 低（CC BY-SA）            | 無 360/俯視覆蓋                    |
| CCTV/單一行車紀錄器       | ❌ 無法 | 無                                                    | 🟢 依來源                    | 單視角無法 SfM                     |
| **無人機/代拍**           | ✅ 極高 | **是，降維打擊**                                      | 🟢 低（你擁版權）            | **唯一解**；地方空拍社團發包數千元 |

### C.2 訓練工具

| 工具                    | 需 GPU    | 戶外品質               | 匯出        | 價格              | 備註                    |
| ----------------------- | --------- | ---------------------- | ----------- | ----------------- | ----------------------- |
| Postshot (Jawset)       | 是        | 極佳、收斂快（分鐘級） | .ply/.splat | 基礎免費/Pro 買斷 | 本地端首選              |
| Luma AI                 | 否(雲)    | 優、天空反光好         | .ply/.splat | 免費/API          | 最無腦                  |
| SuperSplat (PlayCanvas) | 否(Web)   | **只做清理**           | .ply/.splat | 開源              | **必用**：刪路人/移動車 |
| Nerfstudio (Splatfacto) | 是(24GB+) | 極佳、研究級           | .ply        | 開源              | 門檻高                  |
| Scaniverse (Niantic)    | 否(手機)  | 中、大路口會糊         | .ply/.spz   | 免費              | 適合極小範圍            |

### C.3 流程（為 R3F + reconstruction.json 設計）

- **擷取**：iPhone Pro（關 HDR 鎖曝光）最低；最佳＝空拍機 + 地面步行。路徑：地面沿四角繞、
    鏡頭朝路口中心(Orbit)；高空 10–15m 朝下 45° 繞 + 正上方網格。**陰天、避尖峰**。

- **訓練/清理**：抽影格(2–3 FPS, 500–1500 張) → Postshot/Luma → SuperSplat 套索刪破圖/雜物 →
    匯出。

- **格式**：強烈建議 **.spz**（500MB → \<50MB）。轉換：[Niantic SPZ 工具](https://github.com/nianticlabs/spz)。

- **對位（前端可視化法，最適合前端工程師）**：

    1. 先用 reconstruction.json 畫軌跡線當「參考骨架」（0,0 = 路口中心）。
    2. mkkellogg 載 .spz（與 three 0.169 相容）。
    3. 外層包 `<TransformControls>` 或用 `leva` 綁 position/rotation/scale。
    4. **手動拖到 splat 車道線貼合軌跡線** → 把 `{x,y,z,rotX,rotY,rotZ,scale}` 寫死。

    ```jsx
    import { Splat } from '@react-three/drei' // 新版 drei 已封裝 splat
    export function Scene() {
      return (
        <group>
          <Vehicles tracks={reconstructionData} />
          <group position={[12.5, -0.2, -5.1]} rotation={[0, Math.PI/4, 0]} scale={2.35}>
            <Splat src="https://.../intersection-cleaned.spz" />
          </group>
        </group>
      )
    }
    ```

- **效能（重要）**：mkkellogg 用 Web Worker 排序，需伺服器標頭
    `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`
    才能用 SharedArrayBuffer；否則效能掉 5–10×、手機卡死。

    > ⚠️ 本專案注意：`require-corp` 會擋掉跨來源 HDRI/Google tiles，需折衷（SPLAT_NOTES §8.4）。

### C.4 動態場景

- 你的「靜態 splat 背景 + Three.js/GLTF 動態車」是**業界最標準、效能最好**的事故重現做法。
    4D 動態 Splatting（SSTD-GS 等）是「播放錄影」，不適合用 JSON 精確控制每台車——**堅持現有
    架構**。

### C.5 法律/鑑識

- 嚴肅鑑識（呈堂）主流仍是 PC-Crash / Virtual CRASH + Leica LiDAR。3DGS 屬「示意性證據」非
    「測量證據」（SfM 可能幾何微變）。高擬真展示用 3DGS 完美；公釐級煞車痕測量作法律證據仍有
    爭議。

### C.6 仍有爭議

- 反光材質剝離：積水/玻璃反光被烘焙進視角，動態車過去無法正確遮擋倒影。
- 無縫接縫：3DGS 邊緣與 Google tiles(遠景) 的深度遮擋難完美，通常邊緣加 fog 過渡。

連結：[Jawset Postshot](https://www.jawset.com/)、[Luma](https://lumalabs.ai/interactive-scenes)、
[SuperSplat](https://playcanvas.com/supersplat)、[Nerfstudio](https://docs.nerf.studio/)、
[Scaniverse](https://scaniverse.com/)、[Niantic SPZ](https://github.com/nianticlabs/spz)。

---

## D. DeepSeek 報告

### D.1 核心：**不能**。任何「不到現場」畫質都不會優於 Google 3D Tiles，多數涉法律風險

| 來源                           | 可行性                  | 優於 3D Tiles          | 法律            | 實例                     |
| ------------------------------ | ----------------------- | ---------------------- | --------------- | ------------------------ |
| Google Maps/Earth 截圖 → Splat | ⚠️ 學術可行、實務不建議 | 否（俯視為主、牆面糊） | 高（禁衍生 3D） | GBM 論文                 |
| Google Earth Studio 影片       | ⚠️ 同上                 | 否（缺地面視角）       | 高              | 同上                     |
| Google Street View             | ❌ 幾乎不可行           | 否（視差不足、動態多） | 極高（明禁）    | —                        |
| Mapillary                      | ⚠️ 授權好但覆蓋不足     | 視情況                 | 中低            | Mapillary                |
| 行車紀錄器/CCTV                | ❌ 不可行               | 否（無視差）           | 低              | —                        |
| 無人機代拍                     | ✅ 可行                 | 是                     | 低（注意空域）  | Varjo Teleport Autopilot |
| 實地手機/相機                  | ✅ **最佳**             | 是（photorealistic）   | 無              | Luma/Postshot            |

- 論證：Splat 品質上限 = 輸入影像品質 × 視角覆蓋度。Google Earth 是衛星/航拍俯視，缺**地面
    水平視角**與**足夠視差基線** → 路面標線/路緣/號誌桿細節重建不出來。Street View 視差不足、
    動態多、魚眼畸變、ToS 明禁（引 SGD: Street View Synthesis 論文）。
- **真正可行的「不到現場」＝專業無人機代拍**：
    - **Varjo Teleport Autopilot**（2026-05）：地圖畫多邊形 → 自動規劃飛行 → 幾小時回傳 splat。
    - **DJI Terra V5.0+**：RTK 地理座標，直接輸出 georeferenced 3D Tiles/PLY，約 $2,800–4,400。
    - 找當地空拍服務商。

### D.2 工具表

| 工具              | 平台            | 需 GPU       | 戶外品質              | 匯出                 | 價格            |
| ----------------- | --------------- | ------------ | --------------------- | -------------------- | --------------- |
| Luma AI           | iOS/Android/Web | 否(雲)       | ★★★★★                 | PLY/GLB/USD          | 免費            |
| Postshot          | Windows         | 是(RTX3060+) | ★★★★☆                 | PLY                  | 免費Beta/$199yr |
| Polycam           | iOS/Android/Web | 否(雲)       | ★★★★☆（LiDAR 1–2cm）  | PLY/GLTF/FBX/USDZ    | 免費/$150yr     |
| Scaniverse        | iOS/Android     | 是(裝置)     | ★★★★☆                 | .spz                 | 免費            |
| Nerfstudio/gsplat | 跨平台          | 是           | ★★★★★                 | PLY                  | 開源            |
| KIRI Engine       | iOS/Android     | 否(雲)       | ★★★★☆                 | PLY/OBJ              | 免費/付費       |
| Varjo Teleport    | Web             | 否(雲)       | ★★★★★（無人機大範圍） | 私有/API             | 企業            |
| DJI Terra         | Windows         | 是(本地)     | ★★★★★（RTK 地理座標） | 3D Tiles/PLY/GeoTIFF | $2,800–4,400    |

**Web 觀看器**

| 工具                          | three 相容      | 格式                     | COOP/COEP  |
| ----------------------------- | --------------- | ------------------------ | ---------- |
| @mkkellogg/gaussian-splats-3d | ≥0.160          | .ply/.splat/.ksplat/.spz | 需設定     |
| @sparkjsdev/spark             | ≥0.180          | 多種                     | 需設定     |
| PlayCanvas SuperSplat         | 獨立 Web 編輯器 | .ply                     | 瀏覽器直跑 |
| Babylon.js                    | 獨立            | .ply                     | 需設定     |

### D.3 流程

- **最小可行（到現場）**：手機 4K30 慢走 1–3 分鐘（陰天）→ Luma → 下載 PLY → SuperSplat 清 →
    2–3 GCP 對位 → mkkellogg 顯示。
- **最高品質**：空拍機 + 地面 360(Insta360 X4) + 手機互補；鎖 ISO/快門/白平衡；清晨/深夜避動態；
    Nerfstudio/gsplat 或 Postshot 訓練 → PLY → .ksplat → SuperSplat 清 → GCP 對位 → .spz → spark/
    mkkellogg（設好 COOP/COEP）。

### D.4 對位（關鍵）

- **方法一（GCP 相似變換，最可靠）**：路口選 2–3 明顯地物（路燈底座、號誌、路緣轉角）→ 記 GPS
    轉 ENU → 在 splat 找對應點簇 → 求 **7 參數相似變換**（CloudCompare Align/ICP 或 Python
    scipy Procrustes）→ 套用到所有高斯點中心。
- **方法二（直接帶座標）**：DJI Terra(RTK) / Nerfstudio(每張帶 GPS)。

### D.5 法律/鑑識

- Google ToS 明禁大量下載/衍生 3D/逆向工程；用 Google 影像訓練 splat 屬灰色地帶，商業風險高。
- 法庭：有 2026 論文《Accuracy of 3DGS for virtual crime scene reconstruction》定量評估精度；
    傳統 PC-Crash/Virtual CRASH 仍主流，3DGS 需額外驗證幾何精度。建議保留原始素材、GCP 量測、
    處理日誌建立證據鏈。

### D.6 仍無解/爭議

1. 「不到現場」天花板：拿不到地面水平視角 → 路面/路緣/號誌細節永遠補不齊。
2. Google ToS 對「衍生 3D」偏禁、灰色。
3. Splat 法庭採納先例極少。
4. 動態物烤入：佔比過大時事後難完美修。
5. 對位精度：2–3 GCP 達公尺級；公分級(鑑識)需更多 GCP + splat 幾何精度，目前少有工具能自動
    輸出 georeferenced splat（除 DJI Terra 等）。

連結：[Luma](https://lumalabs.ai)、[Postshot](https://postshot.ai)、[Polycam](https://poly.cam)、
[Scaniverse](https://scaniverse.com)、[Varjo Teleport](https://teleport.varjo.com)、
[gsplat](https://github.com/nerfstudio-project/gsplat)、
[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)、
[SuperSplat](https://superspl.at/editor)、
[GBM arXiv 2501.00625](https://arxiv.org/abs/2501.00625)、
[Swyvl 教學](https://swyvl.io/blog/how-to-create-gaussian-splats/)。

---

## E. 三方對照速覽

| 議題                     | Manus                | Gemini                      | DeepSeek                 | 本專案採用                   |
| ------------------------ | -------------------- | --------------------------- | ------------------------ | ---------------------------- |
| 不到現場能否 Forza 級    | 偏可（Earth Studio） | ❌ 不行                     | ❌ 不行                  | **❌ 不行**（採多數）        |
| Google Earth Studio 畫質 | 大幅超越             | 融化、≤tiles                | ≤tiles                   | **≤tiles，不採**             |
| 不到現場的正解           | Earth Studio         | **無人機代拍**              | **無人機代拍/Varjo/DJI** | **無人機代拍**               |
| 架構（靜態+動態疊加）    | ✅                   | ✅                          | ✅                       | ✅                           |
| web viewer               | mkkellogg            | mkkellogg/drei Splat        | mkkellogg/spark          | **mkkellogg**（three 0.169） |
| 對位                     | GCP/SplatTransform   | **拖曳(TransformControls)** | GCP/CloudCompare         | 拖曳 + GCP                   |
| 格式                     | .spz                 | .spz                        | .spz                     | **.spz**                     |
| 鑑識定位                 | 視覺化工具           | 示意證據                    | 示意證據                 | 示意輔助、非測量             |
