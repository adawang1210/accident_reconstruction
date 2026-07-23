# 車禍重建 3D 回放（Three.js / R3F 前端）

獨立於後端 Python 套件的前端。讀取後端輸出的 `reconstruction.json`
（見 [`docs/frontend_api.md`](../docs/frontend_api.md)），在 3D 場景中依**分析出的軌跡與
速度**回放車輛、畫道路中心線、標示撞擊點，並可拉時間軸。

技術選型與調研依據見 [`RESEARCH.md`](./RESEARCH.md)。已含 R3F 外殼 + 資料驅動回放、
glTF 車輛匯入、HDRI/ACES 打光，以及 **Google Photorealistic 3D Tiles** 整合（需 API key
才看得到實景，見下方）。

## 執行

```bash
cd frontend
npm install
npm run sync:scenes    # 從 ../data/ 匯入實際跑過的重建結果（見下）
npm run dev            # 開 http://localhost:5173
```

## 場景資料（實際跑過的路線）

`npm run sync:scenes` 掃描 `../data/**/*_reconstruction.json`——也就是後端 pipeline
每次跑完的產物——複製到 `public/scenes/`，並寫出 `public/scenes/index.json` 清單。
前端開場即載入清單，HUD 左上角可切換場景，也可用網址指定：

```
http://localhost:5173/?scene=keelung_xinwu_yier
```

清單依 `gcp_ground_span_m` 由大到小排序，所以**預設開在校正範圍最大、車速最可信的
場景**。`data/` 只在主 repo（gitignored），所以這個指令要在主 repo 目錄跑。

> `public/scenes/` 同樣在 `.gitignore`，避免誤提交真實事故 GPS；重新 clone 後跑一次
> `npm run sync:scenes` 即可。

### 指定單一來源（略過清單）

設定 `VITE_RECONSTRUCTION_URL` 就會固定載入該來源、隱藏場景選單：

1. **靜態檔**：複製到 `public/reconstruction.json`，設
    `VITE_RECONSTRUCTION_URL=/reconstruction.json`。
2. **後端 API**：啟動 FastAPI 工作台後，設
    `VITE_RECONSTRUCTION_URL="/api/reconstruction?video=<檔名>"`。
    `vite.config.ts` 的 dev proxy 會把 `/api` 轉到 `http://127.0.0.1:8000`。
3. 其他來源：任意設定 `VITE_RECONSTRUCTION_URL` 指向回傳該 JSON 的網址。

## 結構

```
src/
├── types.ts                 # reconstruction.json 型別（對應後端 schema）
├── io/useReconstruction.ts  # 載入 JSON（API 或靜態檔）
├── playback/store.ts        # 全域播放時鐘（zustand）：currentTime / 播放 / 速率
├── scene/
│   ├── sampleTrack.ts        # 依 t_sec 的時間內插（保留真實加減速；核心）
│   ├── Scene.tsx             # 燈光、相機、控制、組裝
│   ├── Ground.tsx            # 占位地面（Phase 2 換 Google 3D Tiles）
│   ├── Roads.tsx             # 道路中心線
│   ├── Vehicle.tsx           # 占位車（Phase 1 換 glTF）+ 沿軌跡移動/轉向
│   └── ImpactMarker.tsx      # 撞擊點標記
└── ui/Timeline.tsx          # 時間軸：播放/暫停、scrub、0.25/0.5/1× 速率
```

## 座標約定

後端的 `x_m`=東、`z_m`=北、原點 `origin_latlon`（ENU 平面）。前端映射到 three.js：

- `x` = `x_m`（東）
- `z` = `-z_m`（北 → -Z，俯視時北朝上）
- `y` = 高度

所有車輛/道路/撞擊點都在同一公尺座標系，直接對齊。

## 回放原理（重點）

車輛位置以 **`t_sec` 時間內插**（`sampleTrack.ts`），**不是**沿曲線弧長等速取點——
這樣才能忠實反映 `speed_kmh` 的加減速。所有車共用一個 `currentTime`，自動同步。
車頭朝向取相鄰點切線，用四元數 `slerp` 平滑。

## 真實環境：Google Photorealistic 3D Tiles（看到那條真實路口）

周遭環境的「實景逼真」靠 Google 3D Tiles——**只需 GPS（`origin_latlon`）、不用實地拍攝**。
需要你自己的 **Google Maps Platform API key**：

1. Google Cloud Console 建專案、**啟用 Map Tiles API**、綁定計費帳號、建立 API key
    （Photorealistic 3D Tiles 屬 Enterprise SKU，有每月免費額度，超過計費）。
2. 在 `frontend/.env` 設 `VITE_GOOGLE_TILES_KEY=你的key`，重新 `npm run dev`。
3. 有 key 且 JSON 有 `origin_latlon` 時，`ReorientationPlugin` 會把實景對到本地原點
    （up=+Y、east=+x、north=-z），道路/車輛/撞擊點直接落在真實地面上；沒有 key 則用占位地面。

> 程式已寫好（`src/scene/GoogleTiles.tsx`），但**真實實景畫面需要你的 key 才能看到/驗證**。

## 匯入車輛模型

預設是程序化占位車。要換成現成模型（「匯入別人做好的」）：把 `.glb`/`.gltf` 放到
`public/models/`，在 `.env` 設 `VITE_CAR_MODEL_URL=/models/car.glb`。
（目前所有車共用同一個模型；之後可依車種 `car`/`motorbike` 分別指定。）

## 後續階段（見 RESEARCH.md）

- **Phase 1（已做）**：HDRI 環境光、ACES tone mapping、即時陰影、`ContactShadows`、
    車輛改 glTF 匯入 + 程序化備援。
- **Phase 2（已寫好程式，待 key 驗證）**：Google Photorealistic 3D Tiles，
    以 `ReorientationPlugin` 對位到 `origin_latlon`。
- **Phase 3（選配）**：重點路口補 Gaussian Splatting（`@sparkjsdev/spark`）；後處理
    （Bloom 車燈 / GTAO）與濕滑路面反射（`MeshReflectorMaterial`）。
