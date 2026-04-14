# BEAT2 到 NAO 的三指标评估重构计划

## 1. 总体原则

论文评估固定只采用 3 个指标：

- `FGD`
- `BC`
- `SRGR`

当前仓库中已经写好的 proxy 版评估代码：

- 保留不动
- 仅作为跑通链路和验证参考
- 不直接作为论文正式实现

后续新写的正式版指标，为了和当前版本区分，代码命名统一在后面加 `_v2`：

- `FGD_v2`
- `BC_v2`
- `SRGR_v2`

---

## 2. FGD 重构方向

参考论文：

- *Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity*

### 2.1 当前遇到的问题

当前 `FGD` 的问题是：

- 现有实现依赖特定动作表示空间中的 encoder
- `SMPL-X npz` 与 `robot pkl` 不属于同一种天然输入分布
- 当前 proxy 版只是把 `pkl` 临时映射成类 SMPL-X 表示以便跑通
- 因此当前结果不能直接作为论文正式指标

### 2.2 需要进一步讨论的问题

新的 `FGD` 需要先回答：

- GT 和 Pred 到底应该映射到什么共同空间
- 这个共同空间是否应该脱离 GMR 当前的 `ik_table`
- 新的 encoder 应该在哪种表示空间中训练

### 2.3 当前确定的方向

准备引入一个抽象的、通用的 layout，用于映射：

- 类人 / 数字人动作
- 机器人动作

使两者进入同一个共享空间后，再考虑新的 `FGD_v2`。

这里的设计思路可参考：

- *Nonparametric Motion Retargeting for Humanoid Robots on Shared Latent Space*

### 2.4 当前结论

`FGD_v2` 暂不立即定稿，先保留为重点讨论项。

---

## 3. BC 重构方向

参考论文：

- *Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation*

### 3.1 原则

`BC` 不像 `FGD` 那样依赖参考分布。

因此：

- `BC_v2` 可以直接重构
- 不必沿用当前 proxy 版的类 SMPL-X 轴角方案

### 3.2 重构目标

新的 `BC_v2` 要严格按照原论文的思路：

- 评估 motion beat 和 audio onset 的一致性

### 3.3 实现要求

评估代码脚本应同时兼容：

- `SMPL-X npz`
- `robot pkl`

也就是说，后续 `BC_v2` 的输入层需要统一封装：

- `human motion -> BC-compatible representation`
- `robot motion -> BC-compatible representation`

然后在同一个节拍评估逻辑中计算最终 `BC` 分数。

---

## 4. SRGR 重构方向

参考论文：

- *BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis*

### 4.1 原则

新的 `SRGR_v2` 准备尽量严格按照原论文的思路重构。

当前仓库里的 `SRGR` 已经能跑通，但它和 GMR 的 `ik_table` 耦合较深。

后续目标是：

- 解耦 GMR
- 单独定义一套与具体机器人无关的共享 landmarks

### 4.2 第一版通用 landmarks

第一版方案先定义如下 canonical landmarks：

- `root`
- `torso`
- `head`
- `left_hand`
- `right_hand`
- `left_elbow`
- `right_elbow`
- `left_knee`
- `right_knee`
- `left_foot`
- `right_foot`

### 4.3 重构方向

`SRGR_v2` 需要：

1. 先将 GT 和 Pred 都映射到上述通用 landmarks
2. 与当前已有 `SRGR` 做对比
3. 去掉对 GMR `ik_table` 的直接依赖
4. 后续步骤再尽量严格按论文中的 SRGR 定义实现

---

## 5. 命名与实现对比方式

为了方便和旧版对比，正式重构版代码统一采用 `_v2` 后缀：

- `FGD_v2`
- `BC_v2`
- `SRGR_v2`

旧版：

- 保留
- 不删除
- 用于链路对照和实验验证

新版：

- 作为论文正式实现方向

---

## 6. Future Work

后续在 `SRGR_v2` 的通用 landmarks 稳定后，可以进一步扩展：

- 在通用 landmarks 基础上加入 `NAO` 机器人的特有 landmarks

目的不是替代通用 landmarks，而是：

- 在保持通用性的前提下，增强对 `NAO` 特定表达能力的覆盖

---

## 7. 当前执行顺序

当前建议的工作顺序是：

1. 保留现有 proxy 指标作为参考
2. 先讨论并确定 `FGD_v2` 的通用 layout
3. 优先重构 `BC_v2`
4. 再重构 `SRGR_v2`
5. 最后回到 `FGD_v2`

这样做的原因是：

- `BC` 和 `SRGR` 不依赖 `FGD` 那种参考分布 encoder 训练问题
- 更适合先完成逻辑上的统一重构
