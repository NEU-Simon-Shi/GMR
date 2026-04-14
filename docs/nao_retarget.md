# GMR 中的 NAO 重定向说明

这份文档面向一个具体问题：

如果你在 GMR 里把 BEAT2 的人体动作重定向到 NAO，再用 FGD、BC、SRGR 去评估，那么这些结果到底在衡量什么，哪些地方是严格映射，哪些地方只是工程上的代理方案。

本文会分四层来讲：

1. GMR 这个项目本身实现了什么。
2. NAO 在 GMR 里的映射方式是什么。
3. 为了让 BEAT2 数据能接到 NAO，这个项目做了哪些适配。
4. 这些设计会如何影响你对 FGD、BC、SRGR 结果的判断。

---

## 1. GMR 这个项目到底实现了什么

GMR 的全名是 General Motion Retargeting。

它的核心目标不是“生成动作”，而是“重定向动作”：

- 输入是人体动作。
- 输出是机器人动作。
- 中间通过一套统一的 IK 目标定义，把人体关键部位的运动映射到不同机器人身上。

通俗地说，GMR 像一个“动作翻译器”：

- 人体这边说的是 SMPL-X、BVH、FBX、Xsens 之类的语言。
- 机器人这边说的是 MuJoCo 模型里的 `qpos`、根位姿、关节角。
- GMR 做的事，就是把前者翻译成后者。

### 1.1 GMR 的通用处理链路

以 `scripts/smplx_to_robot.py` 为例，链路大致是：

1. 读取人体动作文件。
2. 用 SMPL-X body model 解出每一帧的人体关节全局位置和朝向。
3. 初始化 `GeneralMotionRetargeting(src_human="smplx", tgt_robot=...)`。
4. 根据某个机器人对应的 `ik_config`，选出“机器人哪些 body 要去跟人体哪些 body 对齐”。
5. 每帧把这些人体目标点送进 MuJoCo + Mink 的 IK 求解器。
6. 求出机器人每一帧的 `qpos`。
7. 可视化，或者导出成 `pkl`。

对应的关键文件：

- `general_motion_retargeting/motion_retarget.py`
- `general_motion_retargeting/params.py`
- `general_motion_retargeting/utils/smpl.py`
- `scripts/smplx_to_robot.py`

### 1.2 GMR 输出的机器人动作是什么

对评估最重要的一点是：

GMR 导出的机器人动作不是“机器人 body 的位置序列”，而是更底层的状态：

- `root_pos`
- `root_rot`
- `dof_pos`

保存逻辑就在 `scripts/smplx_to_robot.py` 的结尾。

也就是说，GMR 真正输出的是：

- 机器人根节点平移
- 机器人根节点旋转
- 机器人各个自由度的关节位置

这是机器人控制友好的格式，但不是 FGD、BC、SRGR 原始论文直接想吃的格式。

这就是为什么后面评估时还需要额外适配。

---

## 2. GMR 中 NAO 的位置

NAO 在这个项目里已经被当成一个正式支持的目标机器人接入，而不是临时脚本。

### 2.1 NAO 在全局参数中的注册

在 `general_motion_retargeting/params.py` 里，NAO 已经具备三类基础配置：

- MuJoCo 模型：
  `ROBOT_XML_DICT["nao"] = assets/nao/nao_scene.xml`
- SMPL-X 到 NAO 的 IK 配置：
  `IK_CONFIG_DICT["smplx"]["nao"] = general_motion_retargeting/ik_configs/smplx_to_nao.json`
- viewer 跟随主体：
  `ROBOT_BASE_DICT["nao"] = "torso"`

所以从工程结构看，NAO 现在走的不是旁路，而是和 G1、H1、H1-2 一样的主流程。

### 2.2 NAO 的输入支持范围

这里有一个很关键的现实差异：

GMR 原仓库对宇树机器人，尤其是 Unitree G1，支持的输入格式更丰富：

- SMPL-X
- LAFAN1 BVH
- Nokov BVH
- Xsens BVH
- FBX
- XRoboToolkit/PICO

但 NAO 当前在参数表里只注册了：

- `smplx -> nao`

也就是说：

- 宇树机器人在 GMR 里是“多输入格式、成熟主路径”的代表。
- NAO 当前是“SMPL-X 主路径已经通，但其他输入格式还没补”的状态。

这不代表 NAO 不能做更多，而是说明目前你手头最稳定、最清晰的链路是：

`BEAT2 -> SMPL-X 兼容 -> GMR -> NAO`

### 2.3 NAO 的 IK 映射重点

NAO 的 IK 配置在：

- `general_motion_retargeting/ik_configs/smplx_to_nao.json`

核心特点有三点。

#### 第一，NAO 的根节点不是 pelvis，而是 torso

NAO 配置里：

- `robot_root_name = "torso"`
- `human_root_name = "pelvis"`

而 Unitree G1 的配置里：

- `robot_root_name = "pelvis"`
- `human_root_name = "pelvis"`

这意味着 NAO 链路从一开始就不是“骨盆对骨盆”的天然对应，而是“人体骨盆驱动机器人 torso 主体”。

对于评估理解来说，这很重要：

- NAO 的根定义和人体根定义并不完全同构。
- 因此 NAO 的动作更像“主体姿态 + 肢体响应”，而不是完整的人体躯干结构复刻。

#### 第二，NAO 的尺度压缩明显更激进

`smplx_to_nao.json` 里的 `human_scale_table` 比 G1 小得多。

例如：

- NAO 的 `pelvis` 缩放是 `0.32`
- G1 的 `pelvis` 缩放是 `0.9`

这反映了一个很直观的事实：

- G1 是成人尺寸的人形机器人，和人体的比例关系更接近。
- NAO 体型更小，必须把人体目标显著缩小后再送入 IK。

这会导致两个后果：

1. NAO 的动作空间天然更受压缩。
2. 一些大幅度手臂、跨步、躯干动作在 NAO 上会被“压扁”。

所以即使重定向是正确的，NAO 的最终动作幅度也常常不可能和原人体完全一致。

#### 第三，NAO 的匹配 body 数量更少，且位置权重更集中

NAO 当前在 `ik_match_table1/2` 中主要对齐的是：

- `torso`
- 左右髋、膝、踝
- `Head`
- 左右上臂、前臂、末端

它的主体是：

- 躯干
- 双腿
- 双臂

但和一些 DoF 更高、链路更长的宇树机器人相比，NAO 的动作表达能力更有限。

特别是和 Unitree G1 对比时，可以这样理解：

- G1 更接近“完整的人形骨架映射”
- NAO 更接近“保留主体语义的压缩映射”

这会直接影响你怎么读评估指标。

---

## 3. NAO 与宇树机器人对比

这里建议你把 Unitree G1 当成 GMR 里的“成熟基线”。

### 3.1 为什么用 G1 作为对比对象

原因很简单：

- G1 是 GMR 里支持最完整的一批机器人之一。
- G1 既支持 SMPL-X，也支持多种 BVH/FBX/XR 输入。
- 很多脚本默认值就是 G1。
- G1 的配置和项目原始主线最贴近。

所以如果你想判断 NAO 的评估链路是否合理，最好的思路不是问“NAO 和论文是不是一模一样”，而是问：

和 GMR 中已经比较成熟的宇树链路相比，NAO 现在在哪些地方等价，哪些地方是特化处理。

### 3.2 对比总结

| 维度 | Unitree G1 | NAO |
| --- | --- | --- |
| 在 GMR 中的成熟度 | 主线机器人，支持格式多 | 已接入主线，但当前主要走 SMPL-X |
| 根节点语义 | `pelvis -> pelvis` | `pelvis -> torso` |
| 机器人尺度 | 更接近成人人形 | 明显更小，需要强缩放 |
| DoF 与肢体表达 | 更强 | 更受限 |
| 重定向风格 | 更接近“结构保真” | 更接近“语义保真 + 动作压缩” |
| 对评估的影响 | 更容易与人体指标对齐 | 更容易出现“动作合理但数值不完全像人体” |

### 3.3 对评估判断最重要的差异

如果一句话概括：

Unitree G1 更适合被拿去做“接近人体姿态语义的比较”，
而 NAO 更适合被拿去做“共享关键点空间上的功能性比较”。

这也是为什么在你的现有方案里：

- `SRGR` 对 NAO 更有说服力
- `FGD/BC` 对 NAO 更像工程代理指标

---

## 4. 为了把 BEAT2 接到 NAO，做了哪些适配

你的当前链路其实不是“BEAT2 直接喂给 NAO”，而是分了三层。

```text
BEAT2 原始人体动作
-> GMR 兼容的人体动作
-> GMR 输出的 NAO 动作
-> 评估适配
```

下面分别解释。

### 4.1 原始 BEAT2 文件长什么样

原始 GT 一般来自：

- `datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/*.npz`

在 `evaluation/adapters/beat2.py` 里，当前支持两类输入：

1. 原始 BEAT2 风格：
   有 `poses` 和 `trans`
2. GMR 兼容风格：
   有 `pose_body`、`root_orient`、`trans`

也就是说，评估层现在允许你既可以拿原始 BEAT2 `.npz` 做 GT，也可以拿 GMR 转过的 `*_amass_compat.npz` 做 GT。

### 4.2 为什么需要 `*_amass_compat.npz`

这个中间文件由：

- `scripts/beat2_to_robot.py`

生成，默认放在：

- `motion_data/BEAT2/converted/*_amass_compat.npz`

它的作用不是生成机器人动作，而是把原始 BEAT2 整理成更适合 GMR 主链路读取的 SMPL-X 兼容格式。

最重要的转换包括：

- 从 `poses` 拆出 `root_orient`
- 从 `poses` 拆出 `pose_body`
- 保留 `trans`
- 统一 `betas`
- 统一 `gender`
- 统一 `mocap_frame_rate`

另外，针对 raw BEAT2，当前 `SRGR` 还补了一个关键修正：

- raw `poses` 路径按 `Y-up -> Z-up` 处理
- `amass_compat` 文件保持当前管线坐标语义

这一步是为了让 raw GT 和 converted GT 落在同一坐标系里。

### 4.3 NAO 的最终输出文件长什么样

通过：

- `scripts/smplx_to_robot.py`
- 或 `scripts/beat2_to_robot.py`

最终得到：

- `motion_data/BEAT2/retargeted/*_nao.pkl`

这个 `pkl` 里核心字段是：

- `fps`
- `root_pos`
- `root_rot`
- `dof_pos`

这就是 GMR 对 NAO 的标准机器人输出。

注意，这里还不是评估友好的格式。

因为评估模块大多最初是为“人体动作序列”设计的，不是为“机器人状态向量”设计的。

---

## 5. 评估时，BEAT2 和 NAO 是怎样被对齐的

这是本文最重要的部分。

### 5.1 FGD 的适配方式

FGD 原本更接近“人体姿态分布差异”的度量。

在你现在的实现里：

- GT 侧：
  `BEAT2/GMR npz -> SMPL-X axis-angle -> rot6d`
- Pred 侧：
  `NAO pkl -> root_pos + root_rot + dof_pos -> proxy axis-angle -> proxy rot6d`

也就是说，NAO 并没有被恢复成“真正的人体关节旋转”，而是被塞进一个代理的 55 关节 rot6d 张量。

这条路径的意义是：

- 让现有 FGD evaluator 能跑起来
- 能比较不同 NAO 结果之间的相对变化

但它不等价于：

- “NAO 动作在真实 SMPL-X 空间里和人体有多接近”

#### 对 NAO 评估的实际含义

所以对 NAO 来说，FGD 目前应该理解成：

- 一个工程上的代理分布指标
- 可以做相对比较
- 不宜当成严格人体语义距离

#### 为什么 raw GT 和 converted GT 的 FGD 会不同

因为 raw GT 可能包含更完整的 SMPL-X pose 维度，
而 `amass_compat` 只保留 `root + body`，其余部分补零。

因此：

- raw GT 的 FGD
- converted GT 的 FGD

本来就不应该要求严格相等。

所以你在 FGD 上看到 raw 和 converted 的数值不同，不足以说明 NAO 映射坏了。

### 5.2 BC 的适配方式

BC 本来衡量的是动作节拍和音频节拍的一致性。

在你当前实现里：

- GT：
  `BEAT2/GMR npz -> axis-angle -> flatten`
- Pred：
  `NAO pkl -> proxy axis-angle -> flatten`

然后再喂给 `emage_evaltools.BC`。

和 FGD 一样，这里 Pred 侧也不是“真实人体姿态恢复”，而是“机器人状态的代理人体表示”。

但 BC 比 FGD 有个区别：

- BC 只关心速度峰值和节拍响应
- 不太在乎完整手脸姿态语义

这就是为什么你现在看到：

- raw GT BC
- converted GT BC

几乎可以一致。

因为它主要使用的是上半身关节节奏，而 raw 和 converted 在这些核心 body joint 上本来就一致。

#### 对 NAO 评估的实际含义

当前 BC 对 NAO 更像是在回答：

- “这个 NAO 结果有没有跟音频节拍发生响应”

而不是：

- “这个 NAO 是否准确复现了人体的细粒度 gesture beat”

### 5.3 SRGR 的适配方式

SRGR 是目前三者里最适合 NAO 的一个。

原因是它没有把 NAO 状态硬塞回人体姿态槽位，而是改成了共享关键点位置空间。

#### GT 侧

GT 走的是：

`BEAT2/GMR npz -> SMPL-X joints -> 按 GMR 的 NAO IK 配置做缩放与 offset -> 得到人体目标关键点位置`

#### Pred 侧

Pred 走的是：

`NAO pkl -> MuJoCo qpos -> mj_forward -> 取 NAO body 的全局 xpos`

#### 两侧最后怎么比较

两边最终都会投到同一个由 `ik_match_table` 定义的关键点集合上。

例如：

- torso
- 左右脚
- 左右手臂末端

这使得 SRGR 实际比较的是：

- “机器人最终做到的位置”
vs
- “根据同一 IK 语义变换后，人类动作想让机器人做到的位置”

这条定义对于 NAO 非常合理。

因为它避开了 NAO 与人体自由度不一致的问题，转而比较共享任务空间的位置效果。

#### 对 NAO 评估的实际含义

SRGR 现在更接近在回答：

- “NAO 是否把该到的位置做到了”

这比 FGD/BC 当前代理表示的解释性更强。

---

## 6. 现在这三种指标分别该怎么理解

### 6.1 SRGR

对 NAO 最可信。

原因：

- 它比较的是共享关键点位置空间。
- GT 和 Pred 的定义都直接依赖 GMR 的 IK 语义。
- raw GT 和 converted GT 现在已经能得到一致结果。

因此，SRGR 最适合拿来判断：

- NAO 的重定向几何效果是否正确
- IK 目标点是否真的被机器人实现了

### 6.2 BC

中等可信，但要知道它是代理表示。

它适合判断：

- 动作和音频节拍是否有响应关系

它不适合单独判断：

- NAO 是否精准复现了人体 gesture 细节

### 6.3 FGD

当前最弱解释。

不是说不能用，而是它现在更适合做：

- 不同 NAO 方案之间的相对对比
- 调参前后是否变好或变差的参考

不太适合做：

- “NAO 跟人体动作语义距离的严格结论”

---

## 7. 你在判断评估结果时应采用什么标准

建议按下面的优先级来读结果。

### 第一层：先看 SRGR

如果 SRGR 高，而且 raw GT 与 converted GT 一致，说明：

- 坐标系没有错
- GT/Pred 对齐逻辑没有错
- NAO 的关键目标点空间比较是稳定的

这是判断 NAO 重定向链路“是否基本正确”的第一证据。

### 第二层：再看 BC

如果 BC 明显偏低，不要第一时间判定 IK 错。

更合理的解释通常是：

- NAO 动作幅度更小
- NAO 自由度更受限
- 当前 BC 使用的是代理人体表示
- 机器人状态并不天然适合人体节拍速度度量

因此 BC 更适合作为辅助信号。

### 第三层：最后看 FGD

FGD 目前最容易受“表示方式”影响。

尤其要注意：

- raw GT 和 converted GT 的 FGD 不应该混作一个统一标尺
- pkl 侧是代理 rot6d，不是严格人体 rot6d

所以 FGD 更适合看相对变化，不适合单看绝对值做强结论。

---

## 8. 一句话总结

如果你要对 GMR 中的 NAO 重定向和评估做一个最实用的判断，可以这样理解：

- GMR 已经实现了 `BEAT2 -> SMPL-X 兼容 -> NAO qpos` 的完整主链路。
- NAO 在项目中是正式接入的机器人，但相对于宇树机器人，当前支持的输入格式更少，动作表达能力也更受限。
- 对 BEAT2 的适配主要解决了三件事：文件结构统一、坐标系统一、评估输入统一。
- 在现有实现中，`SRGR` 最接近“真实机器人空间评估”，`BC` 次之，`FGD` 当前更像工程代理比较指标。

如果你的目标是判断“NAO 重定向是否正确”，优先信任 SRGR。

如果你的目标是判断“NAO 是否和音频/人体分布更像”，BC 和 FGD 可以参考，但必须始终记住它们现在仍带有代理表示成分。
