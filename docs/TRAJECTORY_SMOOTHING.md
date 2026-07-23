# 軌跡平滑化 — 方法研究與選型

**問題**：目前的平滑（Savitzky-Golay，`ground_footprint.savgol_smooth`）還是會抖、偶爾出現
不合常理的路徑。**根因**：SG 是「局部多項式降噪濾波」，**沒有物理模型**——它不知道車不能
瞬移、不能橫向急跳（jerk）、不能超過轉彎半徑。所以雜訊大或片段短時它仍會 wiggle，也擋不住
違反運動學的軌跡。

學界對「物理上合理的車輛軌跡重建」有明確更好的做法，依對本專案的契合度排序如下。

## 方法一覽

### ① Kalman + RTS 平滑器（本專案採用）
用**運動模型當先驗**：等加速（CA）或等轉率+等加速（CTRA）模型跑前向 Kalman，再跑後向
Rauch–Tung–Striebel（RTS）遞迴。一次同時輸出**平滑的位置＋速度＋加速度**，內建**連續速度/
加速度**約束——這正是「合乎常理」的關鍵。gap 由 predict 步自然帶過；量測雜訊 `r` 與過程雜訊
`q` 兩個參數調平滑度。
- 為何選它：純 NumPy 可寫、無新依賴；線性 CA 模型在 2D 下**過彎不切角**（加速度向量在轉彎時
  指向向心方向，x/y 各自的 CA 模型即可表現轉彎）；輸出帶速度可直接餵下游。
- 來源：[Kalman 平滑演算法](https://www.researchgate.net/publication/371892418_Trajectory_Smoothing_Algorithm_Based_on_Kalman_Filter)、
  [RTS 平滑器原理](https://www.emergentmind.com/topics/rauch-tung-striebel-smoother)、
  [CCTV 車流軌跡擷取用 CA+RTS](https://arxiv.org/pdf/2004.01288)。

### ② 兩步法：先除離群、再平滑（一併採用作前處理）
平滑「之前」先用 **jerk（三階導）或加速度偵測並剔除不合理跳點**，平滑器才不被尖峰帶歪。
剔除的點在 Kalman 裡當「無量測」→ 只 predict。這補強現有的「峰值速度守門」。
- 來源：[兩步法 + 小波](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.13090)、
  [改良離群偵測 + 平滑](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13018/3024008/Two-step-vehicle-trajectory-reconstruction-strategy-based-on-an-improved/10.1117/12.3024008.full)。

### ③ 最小 jerk / 最小曲率 樣條（備選）
擬合一條在容差內貼合觀測、但**最小化 jerk/曲率**的樣條（quintic spline 保證 jerk 連續）。
直接對付「不合常理」——真人開車走最小曲率路徑。與 ① 效果相近但較難處理 gap 與非等時取樣。
- 來源：[Quintic 樣條](https://www.mdpi.com/2032-6653/16/8/434)、
  [最小曲率最佳化](https://arxiv.org/pdf/2309.09186)、
  [限制平滑樣條](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10571921)。

### ④ 車道約束平滑（本專案獨門優勢，待 CAD 線接上）
有 **CAD 車道幾何**後，把軌跡拆成**沿車道（弧長）＋橫向偏移**，只重壓橫向分量（車基本沿
車道走）——橫向抖動被壓、縱向進程不動。最強的物理先驗，一般平滑器做不到。留待
[3D CAD 線](3D_RECONSTRUCTION.md)接上後當最後一層。
- 來源：[車道約束路徑平滑](https://link.springer.com/article/10.1186/s43067-025-00272-3)、
  [人類偏好路徑模型](https://link.springer.com/article/10.1007/s42154-023-00259-8)。

## 怎麼量「是否變合理」（客觀指標）
研究界用 **jerk（三階導）大小、jerk 變號次數、|jerk|>15 m/s³ 的比例**衡量物理合理性
（有論文把 raw 的 jerk 從 ±4900 壓到 ±45）——[加速度去噪](https://onlinelibrary.wiley.com/doi/10.1155/2023/2661136)。
本專案用「平均 |jerk|、最大 |jerk|、jerk 變號次數」比較 SG vs Kalman-RTS。

## 採用與實作
`accident_reconstruction/trajectory_smoothing.py`：
- `kalman_rts_smooth`：2D 等加速 Kalman + RTS 平滑（per-axis，處理非等時 dt 與 gap）。
  接進 `auto_reconstruct.refine_metric_from_contours` 第 3 層，取代 Savitzky-Golay。
- `trajectory_jerk`：驗收指標（平均/最大 |jerk|、變號次數、frac>15）。
- `reject_kinematic_outliers`：獨立診斷工具（硬加速度上限）。**不被平滑器使用**。

### 實作中踩到、修掉的坑
1. **速度初始化**：原本 velocity 初始化為 0，快速物體會嚴重延遲；改用**前幾點最小平方
   擬合斜率**（對雜訊穩健，單點差分在首步小於雜訊時連正負號都會錯）。
2. **離群剔除方式**：不能對「原始位置的二次差分加速度」設門檻——像素雜訊微分兩次會讓
   幾乎每個點看起來都是「超大加速度離群」（實測在測試軌跡上砍掉 60 點中的 58 點、把路徑
   壓平）。改成 **Kalman innovation gating**：只有當量測值離模型預測超過 `gate_sigma`（5σ）
   才跳過該點——這才是正確做法（抓真跳點、留正常雜訊）。

### 實測（BMW，在地輪廓 anchor 原始軌跡上）
指標 `frac>15` = |jerk|>15 m/s³ 的幀比例（越低越合乎常理）：

| | jerk 平均 | frac>15 | 路徑位移 |
|---|---|---|---|
| Savitzky-Golay 汽車 | 216 | **92%** ❌ | 1.7 cm |
| **Kalman+RTS 汽車** | **3.6** | **5%** ✅ | 6.4 cm |
| Savitzky-Golay 機車 | 480 | **91%** ❌ | 2.5 cm |
| **Kalman+RTS 機車** | **3.1** | **0%** ✅ | 12.7 cm |

SG 只壓振幅、**九成的幀仍不合運動學**；Kalman+RTS 把 jerk 降約 60–150 倍、`frac>15` 從 92%
降到 5%（機車 0%），路徑僅移數公分（忠實不失真），且**過彎不切角**（合成正弦曲線 err 0.007，
低於雜訊底 0.033）。

**已知小殘留**：管線最後的**空缺補值**（`interpolate_straight_gaps`）是在 Kalman 之後插入
線性段，與已平滑的曲線段在邊界不 jerk-連續，讓機車最終 `frac>15` 從 0% 回到約 11%（仍遠優於
SG 的 91%）。日後可在補值後再做一次輕量 Kalman 收尾消除邊界。
