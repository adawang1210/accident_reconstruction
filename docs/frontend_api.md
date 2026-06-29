# 前端串接：`reconstruction.json`（給 Three.js 等 3D/Web 視覺化）

整個 2D 重建結果**封裝成單一 JSON**，前端一次取得即可畫車、畫路、跑時間軸動畫，
**不需解析 CSV、不需自己做投影或座標換算**。

## 取得方式（二擇一）

- **HTTP（web 工作台）**：`GET /api/reconstruction?video=<檔名>`
    回傳已 parse 的 JSON。pipeline 跑過該片後才有（否則 404）。
- **靜態檔（離線打包）**：每次 pipeline 跑完寫在
    `data/<scene>/<scene>_reconstruction.json`，可直接打包進前端。
    也可程式產生：`recognized_route.write_reconstruction_json()` /
    `build_reconstruction()`（回傳 dict）。

## 座標約定（關鍵）

所有位置共用**同一個本地公尺平面**（原點 = `origin_latlon`）：

- `x_m` = 東向公尺、`z_m` = 北向公尺。
- 車輛、道路、撞擊點都在這個平面上，**直接對齊**。
- 同時附 `lat`/`lon`（要疊真實地圖時用）。

Three.js 建議對應（地面為 X–Z、Y 向上、俯視北上）：

```js
const toVec3 = p => new THREE.Vector3(p.x_m, 0, -p.z_m); // 北→-Z 讓俯視時北朝上
```

## JSON Schema

```jsonc
{
  "scene": "keelung_xinwu_yier",
  "ready": true,                  // false 時附 "reason"（如尚未校正）
  "fps": 29.0,
  "impact_frame": 153,
  "axes": "x_m=east, z_m=north (metres, north-up ground plane)",
  "origin_latlon": [25.1341019, 121.7474411],
  "impact": {                     // 可能為 null（場景未設真實撞擊點）
    "frame": 153, "lat": …, "lon": …, "x_m": 4.29, "z_m": -1.31
  },
  "vehicles": {
    "taxi": {
      "name": "計程車",
      "color_rgb": [255, 193, 7],
      "road": "義二路",
      "track": [
        { "frame": 120, "t_sec": 4.138,
          "x_m": 1.757, "z_m": -5.242, "lat": …, "lon": …,
          "speed_kmh": 0.0, "is_impact": false },
        …
      ]
    }
  },
  "roads": {                      // 可能為 {}（場景未提供道路中心線，如 BMW）
    "taxi": [ { "x_m": …, "z_m": …, "lat": …, "lon": … }, … ]
  },
  "speed_reliability": { "gcp_ground_span_m": 22.8 }
}
```

欄位說明：

- `vehicles[].track` 依 `frame` 遞增；`t_sec = frame / fps` 已算好，可直接驅動動畫。
- `is_impact` 標出撞擊幀（每車最多一個 true）。
- `speed_kmh` 由**對齊後路徑**用 haversine 重算（與顯示路徑一致）。
- `speed_reliability.gcp_ground_span_m`：GCP 真實涵蓋範圍。**偏小（如 ~18 m）時車速
    會被低估**，前端可據此標註「速度為估計值」。詳見 `docs/summary.md`。

## 最小串接範例

```js
const data = await (await fetch(`/api/reconstruction?video=${name}`)).json();
if (!data.ready) throw new Error(data.reason);

// 畫道路
for (const pts of Object.values(data.roads)) {
    const geo = new THREE.BufferGeometry().setFromPoints(pts.map(toVec3));
    scene.add(new THREE.Line(geo, roadMat));
}

// 每車一條軌跡 + 一個沿時間移動的 mesh
for (const [id, v] of Object.entries(data.vehicles)) {
    const color = new THREE.Color(...v.color_rgb.map(c => c / 255));
    // v.track[i].t_sec / x_m / z_m / speed_kmh …
}
```

## 注意

- 場景沒有道路中心線（`roads: {}`）或真實撞擊點（`impact: null`）時，端點仍正常回傳
    車輛軌跡——前端要對這兩者做 null 處理。
- `ready: false` 代表該片尚未做 GPS 校正；先在工作台校正並重跑。
