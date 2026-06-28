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
    - **修法**：`_segment_masks` 改成**每段建立全新 predictor**，各 re-seed 真正獨立。
        代價是每段重新載入權重（稍慢），但正確。

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

## 工作流程陷阱（重要）

- `.venv` 以 PEP 660 editable 安裝把 `accident_reconstruction` 指向**主 repo 的
    master**。worktree 的程式改動**必須先併進 master** 才會在 `python -m ...` 執行；
    `PYTHONPATH` 不會覆蓋 editable finder。
- `data/` 是 gitignored、只存在於主 repo；跑 pipeline 要在主 repo 目錄。

---

## 相關 commit

- `merge_suppression_cuts` / `box_containment`（模式一）
- `truncate_boxes_at_impact`（模式二自動解）＋ web_app 保留白名單
- merge 閘門 anchor-aware（手動框不被砍）
- backtrack 閘門豁免手動框 ＋ `split_overlapping_masks`（模式三）
- 每段建立全新 SAM2 predictor（修 predictor 重用洩漏記憶，模式三真正主因）
