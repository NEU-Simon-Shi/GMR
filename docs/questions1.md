# 当前问题总览：数字人数据、机器人数据与三类评估指标

## 1. Overview

目前最核心的问题，不是单个指标公式本身，而是：

- 现有评估指标通常可以评价双方都是数字人 / 人体动作数据的情况
- 或者评价双方都是机器人动作数据的情况
- 但难以直接评价“一方面是数字人数据，另一方面是机器人数据”之间的差距

换句话说，当前最大的根本矛盾是：

- **数字人数据和机器人数据不统一**

在当前论文场景中，这个问题表现为：

- GT 是 `BEAT2 SMPL-X npz`
- Pred 是机器人 `pkl`

两者的数据形式、语义、运动学结构都不同：

- `SMPL-X npz` 对应人体骨架与人体动作语义
- `robot pkl` 对应机器人状态空间与机器人运动学

因此，评估的核心挑战不是“怎么把公式抄过来”，而是：

- 如何让人体 / 数字人数据与机器人数据进入可比较的表示空间

当前所有关于 `FGD / BC / SRGR` 的问题，本质上都可以追溯到这里。

---

## 2. FGD

参考论文：

- *Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity*

### 2.1 官方代码 / 官方思路中的 FGD 运行逻辑

FGD 的核心不是直接比较骨骼坐标，而是：

1. 将 motion 输入转换成某种固定动作表示
2. 送入一个预训练 encoder
3. 得到 latent feature
4. 在 feature 分布上计算 Fréchet distance

也就是说，FGD 的整体流程是：

```text
motion sequence
-> feature representation
-> encoder
-> latent features
-> mean / covariance
-> Fréchet distance
```

### 2.2 当前 BEAT / EMAGE 风格 FGD 的数据转换逻辑

在当前项目和 BEAT / EMAGE 风格实现里，FGD 实际要求的输入是：

- `55-joint SMPL-X axis-angle`

然后再转成：

- `55 * 6 = 330` 维的 rot6d

当前 GT 侧的逻辑是：

```text
BEAT2 / GMR npz
-> canonicalize
-> (T, 55, 3) axis-angle
-> rot6d
-> FGD encoder
```

这意味着：

- GT 侧是天然符合现有 FGD 输入假设的

### 2.3 当前临时代码版本的机器人侧逻辑

当前项目中的临时代码，对机器人 `pkl` 的处理并不是真正恢复成人体姿态，而是：

1. 读取机器人 `pkl`
    - `root_pos`
    - `root_rot`
    - `dof_pos`
2. 将：
    - `root_pos(3)`
    - `root_rotvec(3)`
    - `dof_pos(D)`
      拼成一个状态向量
3. 将这个状态向量：
    - 截断或补零到 `165` 维
4. reshape 成：
    - `(T, 55, 3)`
5. 再转成 rot6d 后送入现有 FGD encoder

也就是：

```text
[root_pos(3), root_rotvec(3), dof_pos(D)]
-> truncate / pad to 165 dims
-> reshape(T, 55, 3)
-> rot6d
-> FGD encoder
```

这个做法的目的只是：

- 让现有 FGD evaluator 跑通

它的问题也很明确：

- 机器人状态并没有自然的人体关节语义
- `(T, 55, 3)` 只是人为凑出来的临时表示
- 因此这个临时代码不能直接作为论文正式版 FGD

### 2.4 FGD 当前遇到的关键问题

FGD 目前最大的几个问题是：

1. `SMPL-X npz` 和 `robot pkl` 不属于同一种天然输入分布
2. 当前机器人侧输入只是一套临时补零 / 截断方案
3. FGD 必须依赖一个参考分布
4. 当前还没有一套明确、通用的人体 / 机器人共享表示空间

### 2.5 FGD 需要的数据

要真正重构 FGD，至少需要：

- `BEAT2 SMPL-X npz`
- 一批 `robot pkl`
- **一套共同表示空间**
- **一套新的 encoder 训练数据与训练方案**

### 2.6 FGD 当前的结论

FGD 是三个指标里目前问题最多的一个：

- 临时代码可运行
- 但正式方案仍待进一步讨论

---

## 3. BC

参考论文：

- *Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation*

### 3.1 BC 的原理与边界

BC 的本质是：

- 比较 `motion beats` 和 `audio onsets`

因此：

- BC 不需要参考分布
- BC 可以对每个 motion 单独打分

所以理论上可以分别算：

- `BC(GT npz, wav)`
- `BC(GMR pkl, wav)`
- `BC(YourMethod pkl, wav)`

### 3.2 BC 当前的核心问题

#### 问题 1：当前机器人侧输入仍然是 proxy axis-angle

现有 `BC` 的问题不在评分公式，而在输入表示：

- `npz` 侧：人体 `(T,55,3)` axis-angle
- `pkl` 侧：fake `(T,55,3)` 临时 axis-angle

这意味着机器人侧输入表示并不真实。

#### 问题 2：原有 BC 代码里关于人体数据的变量较多，不太能够直接复用

当前 BC 代码大量建立在人体数据表示之上，例如：

- 输入默认按人体 `(T, 55, 3)` 组织
- 关节索引按人体 skeleton 语义写死
- 归一化统计文件也来自人体 motion

因此：

- BC 不能像“只改一行输入”那样直接复用
- 需要对输入层、索引层和归一化层做较大范围重构

#### 问题 3：当前 BC 代码只选择上半身关节作为输入，是否沿用这一选择

当前实现中：

- BC 并不是对全部关节平均
- 而是只取 upper-body joints 参与 motion beat 分析

这对应着原始 co-speech gesture 任务的直觉，因为：

- gesture 主要由手臂、上半身承担

仍需进一步确认：

- 是否继续沿用“只看上半身”这一选择
- 如果沿用，机器人侧上半身通道应如何定义

#### 问题 4：输入文件 `mmae` 之一，是人体 motion 的 mean velocity 文件，能否直接复用到机器人上

当前 BC 代码里：

- `mmae` 是人体 motion 统计得到的 mean velocity 文件

因此当前需要明确：

- 这份 `mmae` 能否直接用于机器人 `(T, J, 3)` motion
- 如果不能，是不是要重新统计机器人版 `mmae`
- 或者干脆去掉这一步归一化

### 3.3 BC 需要的数据与前提

要重构 `BC`，至少需要：

- `BEAT2 SMPL-X npz`
- `BEAT2 wav`
- 一批 `robot pkl`
- 能从 `pkl` 恢复的 robot 3D trajectories
- 一组为机器人定义的 `upper_body / expressive carriers`

如果要保留 `mmae` 归一化，还需要：

- 一批机器人 motion 数据来统计 robot-specific `mmae`

### 3.4 当前已明确的解决方向

BC 当前最清晰、最稳的重构方向是：

- `npz` 侧逻辑基本不动
- `pkl` 侧不再走 proxy axis-angle
- 改成：
    - `pkl -> Mujoco -> real (T, J, 3)`
- 后续继续复用原 BC 的主体计算框架：
    - 速度
    - 峰值
    - onset matching
    - averaging

这条路线是当前最有可执行性的。

### 3.5 BC 的潜在解决方案

#### 方案：不要 `mmae`

思路：

- 不使用 `mmae`
- 直接基于原始速度范数提取 beats

**目前正在基于此思路修改代码，尚未遇到不确定的问题**

优点：

- 实现简单

问题：

- 不同点的速度尺度可能不均衡

### 3.6 当前关于 BC 的未决问题

- `pkl` 侧最终取哪些 body / site 作为 `J`
- `upper_body` 如何定义
- 是否保留 `mmae`
- 如果保留，机器人版 `mmae` 用哪批数据统计
- `BC` 是否同时支持单文件和目录模式

---

## 4. SRGR

参考论文：

- *BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis*

### 4.1 当前临时代码版本的运行逻辑

当前仓库中的 `SRGR` 临时代码，整体逻辑是：

#### GT 侧

```text
BEAT2 npz
-> SMPL-X body model
-> 得到人体 joints / body frames
-> 通过 GMR 的对齐逻辑做人体现实空间调整
-> 按 ik_table 取对应关键点
-> 得到 (T, J, 3)
ps: ik_table 来自于 GMR 中
```

这里用到的 GMR 对齐逻辑包括：

- `human_scale_table`
- `pos_offset`
- `rot_offset`
- ground alignment

#### Pred 侧

```text
robot pkl
-> qpos
-> mj_forward
-> 取 ik_table 对应 robot body 的 xpos
-> 得到 (T, J, 3)
```

#### 最后打分

两边统一到同一个关键点集合后：

- 对每帧每点算欧氏距离
- 小于阈值记成功
- 如果有 semantic 权重，则再加权。**默认没有**
- 最终平均得到 `SRGR`

### 4.2 当前 SRGR 的关键问题

当前版本的 SRGR 最大的问题不是公式，而是：

- **它依赖于 GMR 的 `ik_table`**

这意味着：

- 当前共享比较空间其实是 GMR-specific 的
- 它和 baseline 的内部设计仍然耦合较深

因此：

- 当前 `SRGR` 可以作为一个合理的临时版机器人化实现
- 但还不是完全解耦后的通用版论文指标

### 4.3 当前 SRGR 的意义

当前版本的 SRGR 在数学形式上仍然和原论文一致：

- 本质上仍然是带 semantic 权重的 PCK

但在输入空间上，它已经变成：

- 基于 GMR `ik_table` 的机器人共享点空间

### 4.4 SRGR 需要的数据

要继续重构 `SRGR_v2`，至少需要：

- `BEAT2 SMPL-X npz`
- `robot pkl`
- 通用共享 landmarks
- `SMPL-X -> landmarks` 映射
- `robot -> landmarks` 映射
- 可选 semantic 文件

### 4.5 SRGR 的当前结论

当前 `SRGR` 的问题比 FGD 少，但比 BC 更深：

- 公式基本没问题
- 主要问题是共享比较空间目前仍依赖 GMR 的 `ik_table`

---

## 5. 未来解决方案

### 5.1 方案一：定义一套与具体机器人无关的通用 landmarks

未来最直接、最清晰的方向，是定义一套和具体机器人无关的 landmarks，例如：

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

然后分别定义：

- `SMPL-X -> 通用 landmarks`
- `robot -> 通用 landmarks`

这样就能把人体数据和机器人数据映射到统一空间。

这条路线最适合：

- `SRGR`
- 潜在的新 `FGD`

### 5.2 方案二：学习 shared latent space

除了显式定义 landmarks，也可以考虑之前提到的 shared latent space 方向。

可参考：

- *Nonparametric Motion Retargeting for Humanoid Robots on Shared Latent Space*

这类方法的核心思路是：

- 不强行要求人体与机器人在显式点位上完全一一对应
- 而是将二者学习到同一个共享潜空间

优点：

- 更灵活
- 更适合跨 embodiment

问题：

- 可解释性相对较弱
- 训练与实现复杂度更高

## 6. 当前最重要的结论

1. 目前最本质的问题是：
    - 数字人 / 人体数据与机器人数据不统一
2. `FGD` 问题最大，因为它不仅要处理输入统一，还要处理参考分布和 encoder 问题
3. `BC` 相对最容易先落地，因为它只关心 `motion vs audio`
4. `SRGR` 当前已经较接近目标，但它仍然依赖 GMR 的 `ik_table`
5. 长期最合理的解决方向是：
    - 定义通用 landmarks
    - 或者学习 shared latent space
