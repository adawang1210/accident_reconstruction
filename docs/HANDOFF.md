# 交接 / 續作摘要（resume here）

新視窗或新 session 接手時看這份。最後更新：2026-06-30。所有內容已在 `master`。

專案目標：把車禍影片重建成**地圖上的 2D 行車軌跡**（後端），並用 **Web 3D**（前端）把
真實路口還原、讓車輛依分析出的軌跡＋速度回放、標示撞擊點。

---

## 1. 目前狀態

### 後端（Python 套件 `accident_reconstruction/`）

- **碰撞後框合併**三種失敗模式已修：小框塌進大框（`merge_suppression_cuts`）、大框吞掉
    融合車（`truncate_boxes_at_impact` / `split_overlapping_masks`）、手動補框失效
    （predictor 重用洩漏記憶 + backtrack 閘門 + anchor-aware）。詳見 `docs/summary.md`。
- **效能**：追蹤 9 分 → ~3.5 分（重用 predictor＋段長截斷）。
- **車速**：診斷出「GCP 涵蓋範圍小 → 速度被低估」（非 bug，需更廣的 GCP 校正）；另修了
    「速度沒套對齊 scale」的 bug。`auto_reconstruct` 會印速度可信度提醒。
- **對前端封裝**：`build_reconstruction()` → 單一 `reconstruction.json`（公尺座標軌跡＋速度＋
    道路中心線＋撞擊點＋fps）。取得：`GET /api/reconstruction?video=<名>`，或檔案
    `data/<scene>/<scene>_reconstruction.json`。schema/座標約定見 `docs/frontend_api.md`。

### 前端（`frontend/`，獨立 Vite + React + R3F app）

- **資料驅動回放**：依 `t_sec` 時間內插（保留加減速）、車頭朝向切線、時間軸 scrub /
    播放 / 0.25–1× 速率。核心：`src/scene/sampleTrack.ts`。
- **車輛**：程序化占位車 + glTF 匯入（`VITE_CAR_MODEL_URL`）。HDRI（drei `<Environment>`）
    - ACES tone mapping + 即時陰影 + `ContactShadows`。
- **底圖（優先序）**：Gaussian Splat（`VITE_SPLAT_URL`）> Google 3D Tiles
    （`VITE_GOOGLE_TILES_KEY`，**已驗證能載入台北實景**）> 占位格線地面。
    - Google Tiles：`ReorientationPlugin` 對位到 `origin_latlon`。已知雷：地面高度/session
        token/COOP-COEP——細節見 `frontend/SPLAT_NOTES.md` 與另一 worktree 的
        `SCENE_NOTES.md`。
    - Splat viewer（mkkellogg `DropInViewer`）已實作、驗證能正確載入；等真實 `.spz`。

---

## 2. 關鍵文件

| 檔                                   | 內容                                             |
| ------------------------------------ | ------------------------------------------------ |
| `docs/summary.md`                    | 後端框合併問題的完整根因與修法                   |
| `docs/frontend_api.md`               | `reconstruction.json` schema、座標約定、串接範例 |
| `frontend/README.md`                 | 前端怎麼跑、env 變數、結構                       |
| `frontend/RESEARCH.md`               | Three.js 底圖路線的四方調研與決策                |
| `frontend/SPLAT_NOTES.md`            | 高斯噴濺決策＋實作（§8 三方整合）                |
| `frontend/SPLAT_RESEARCH_SOURCES.md` | 三份 AI 原始研究存檔                             |
| `frontend/.env.example`              | 前端 env 範本（複製成 `.env` 填值）              |

---

## 3. 環境 / 陷阱（務必知道）

- **`.venv` 跑的是主 repo 的 master**：editable 安裝把 `accident_reconstruction` 指向主 repo
    checkout。**worktree 的 Python 改動要先併進 master 才會生效**；`PYTHONPATH` 不會覆蓋。
    `data/` 是 gitignored、只在主 repo，跑 pipeline 要在主 repo 目錄。（見
    `AGENTS.md §4.5`。）
- **前端 `.env`**（含 Google API key）在 worktree 與主 repo **各一份**、皆 gitignored；
    Vite 啟動時讀，改了要重啟 dev server。
- **`overrides.json` 白名單**：`web_app.save_overrides` 只保留白名單 key，新增 override key
    要同步加進去（已含 `truncate_boxes_at_impact`）。
- 提交前：`.venv/bin/python -m pytest` ＋ `uv run pre-commit run --all-files` 必須過。
    codespell 對某些英文縮寫會誤判（例：層級細節的縮寫）→ 改寫成中文即可。

---

## 4. 下一步（待辦）

1. **取得真實路口 splat**：到現場手機/空拍實拍，或無人機代拍（Varjo Teleport / DJI Terra
    可直接輸出帶座標的 splat）。Google 截圖**不可行**（見 `SPLAT_NOTES.md §1`）。
2. 產 `.spz` → 放 `frontend/public/` → `.env` 設 `VITE_SPLAT_URL` → `npm run dev`。
3. **對位**：用 `VITE_SPLAT_SCALE/ROT_*/X/Y/Z` 或加一個 dev-only `<TransformControls>`
    拖曳，把 splat 路面對到車輛軌跡（見 `SPLAT_NOTES.md §5`、§8.3）。
4. （可選）前端 viewer 目前在 `priceless-euclid-8a3ca5` worktree；另有更進階的
    `stoic-dijkstra-b59875` worktree（含 OSM 示意底圖 + 修好的 tiles）。**要先決定哪個
    frontend 是正本**，避免分叉。

---

## 5. 怎麼跑

```bash
# 後端 pipeline（在主 repo 目錄）
ACCIDENT_SCENE=<scene> .venv/bin/python -m accident_reconstruction.run_pipeline

# 後端 web 工作台
.venv/bin/python -m accident_reconstruction.web_app          # :8000

# 前端
cd frontend && npm install && npm run dev                    # :5173
```
