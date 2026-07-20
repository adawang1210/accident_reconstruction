# Agent 指南 — `accident_reconstruction`

本專案把車禍影片重建成地圖上的 2D 行車軌跡。以資深貢獻者的標準工作：精準、有效率、
重視可維護性與清晰度。

---

## 1. 動手前

- 完整讀懂任務再行動；缺資訊時，問**一個**最關鍵的澄清問題。
- 先列出步驟計畫再改 code。
- 確認功能是否已存在於 `accident_reconstruction/`（避免重複造輪子）。

---

## 2. 專案結構

```
accident_reconstruction/             # 可 import 的套件（純程式碼）
├── scene_config.py                  # 場景設定、路徑、GPS 錨點（SceneConfig / SCENES）
├── calibrate_homography.py          # GCP 校正、ViewTransformer、metric_to_latlon
├── prompt_track_accident.py         # Stage 1：SAM2 視訊記憶追蹤
├── auto_reconstruct.py              # Stage 2：投影 + 撞擊偵測 + 截斷
├── run_pipeline.py                  # 串接 Stage 1 + Stage 2
├── birdseye_manual_annotation.py    # 地圖輸出（KML / 圖片 / CSV）+ 道路對齊
├── recognized_route.py              # 原始投影輸出（無道路對齊）
├── manual_pre_impact_motorcycle_annotation.py  # 永康場景 legacy 手動標記
└── web_app.py / web_app.html        # FastAPI Web 工作台

docs/                                # 文件（已移出套件）
└── README.md / PROJECT_SUMMARY.md / ACCIDENT_2D_RECONSTRUCTION.md
```

- 用 `ACCIDENT_SCENE` 環境變數選擇場景（見 `scene_config.SCENES`）。
- 所有 ML / pipeline 指令用 `.venv/bin/python -m accident_reconstruction.<module>`。

---

## 3. 程式風格

- **格式與 lint** 由 pre-commit 強制（ruff-check、ruff-format、codespell 等）。
- **型別註記**：新程式碼一律加上。
- **Docstring**：Google 風格，新函式/類別必備；範例盡量用原始值，能當可執行文件。
- 中文 UI / CLI 字串刻意使用全形標點（ruff RUF001/2/3 已在 `accident_reconstruction/*.py` 忽略）。

---

## 4. 慣例

- 沿用既有命名與 API；非必要不破壞既有介面。
- 場景相關資料一律放進 `SceneConfig`，**不要**在 stage 模組裡寫死特定場景。
- CSV 是 stage 之間的資料交換格式（`frame, vehicle, anchor_x, anchor_y`）。
- 偏好向量化的 NumPy / OpenCV 運算；避免不必要的陣列複製。

---

## 4.5 陷阱（踩過的雷）

- **`.venv` 跑的是主 repo 的 master**：venv 以 PEP 660 editable 安裝把
    `accident_reconstruction` 指向**主 repo checkout**。worktree 的改動**必須先併進
    master** 才會在 `python -m accident_reconstruction.*` 生效；`PYTHONPATH` 不會覆蓋
    editable finder。`data/` 是 gitignored、只在主 repo，跑 pipeline 要在主 repo 目錄。
- **速度 = homography metric 距離 / 時間**：只有在 GCP 真實涵蓋範圍夠大（涵蓋車輛
    行經的整段路）時才準。GCP 擠在小範圍（如某些場景 ~18 m）時，殘差雖小但車速會被
    嚴重低估。詳見 `docs/summary.md`。

---

## 5. 修 bug

1. 重現並理解根因。
2. 先寫一個能重現 bug 的測試（修之前應該失敗）。
3. 套用最小、精準的修正。
4. 確認測試通過且沒有破壞其他元件。

---

## 6. 提交前

```bash
.venv/bin/python -m pytest          # 若有測試
uv run pre-commit run --all-files
```

- 所有 pre-commit hook 必須通過。
- 修正所有回報的問題後再次執行直到乾淨。
