# 車禍重建 3D 回放（Three.js / R3F 前端）

獨立於後端 Python 套件的前端。讀取後端輸出的 `reconstruction.json`
（見 [`docs/frontend_api.md`](../docs/frontend_api.md)），在 3D 場景中依**分析出的軌跡與
速度**回放車輛、畫道路中心線、標示撞擊點，並可拉時間軸。

技術選型與調研依據見 [`RESEARCH.md`](./RESEARCH.md)。本資料夾是 **Phase 0**：R3F 外殼 +
資料驅動回放 + 占位地面，**尚未接 Google 3D Tiles**（座標與結構已預留）。

## 執行

```bash
cd frontend
npm install
npm run dev            # 開 http://localhost:5173
```

預設載入內建的合成範例 `public/reconstruction.sample.json`，`npm run dev` 即可看到
兩台車交叉、撞擊點閃爍。

## 換成真實資料

`reconstruction.json` 由後端每次 pipeline 跑完產生。三種接法：

1. **靜態檔**：把某場景的 `<scene>_reconstruction.json` 複製到
    `public/reconstruction.json`，並設定
    `VITE_RECONSTRUCTION_URL=/reconstruction.json`（`.env` 或啟動時帶入）。
    （`public/reconstruction.json` 已在 `.gitignore`，避免誤提交真實事故 GPS。）
2. **後端 API**：啟動 FastAPI 工作台後，設定
    `VITE_RECONSTRUCTION_URL="/api/reconstruction?video=<檔名>"`。
    `vite.config.ts` 的 dev proxy 會把 `/api` 轉到 `http://127.0.0.1:8000`。
3. 其他來源：任意設定 `VITE_RECONSTRUCTION_URL` 指向回傳該 JSON 的網址。

## 結構

```
src/
├── types.ts                 # reconstruction.json 型別（對應後端 schema）
├── data/useReconstruction.ts# 載入 JSON（API 或靜態檔）
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

## 後續階段（見 RESEARCH.md）

- **Phase 1**：HDRI 環境光、車輛換 glTF、後處理（Bloom/GTAO）、濕滑路面反射。
- **Phase 2**：接 Google Photorealistic 3D Tiles（`3d-tiles-renderer/r3f`），
    把整個場景群組對位到 `origin_latlon` 的 ENU frame。
- **Phase 3（選配）**：重點路口補 Gaussian Splatting（`@sparkjsdev/spark`）。
