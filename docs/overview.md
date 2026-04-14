# GMR 项目概览

## 1. 项目定位

GMR（General Motion Retargeting）是一个用于**人体动作到机器人动作重定向**的 Python 项目。  
核心能力是把 SMPL-X / BVH / FBX / Xsens 等来源的人体动作统一为关节位姿，再通过 IK 求解映射到不同 humanoid 机器人。

项目特点：

- 以 `general_motion_retargeting` 包为核心，脚本在 `scripts/` 下完成具体流程。
- 依赖 MuJoCo + Mink + qpsolvers 进行逆运动学求解与可视化。
- 通过 `ik_configs/*.json` 调整不同机器人和数据格式的映射关系。
- `assets/` 下包含大量机器人模型、网格和仿真资源。

---

## 2. 项目结构速览

- `general_motion_retargeting/motion_retarget.py`  
  核心类 `GeneralMotionRetargeting`，负责加载机器人模型、IK 配置、更新人体目标、迭代求解 IK 并输出 `qpos`。
- `general_motion_retargeting/params.py`  
  维护机器人模型路径、IK 配置路径、viewer 跟随主体等全局映射表。
- `general_motion_retargeting/robot_motion_viewer.py`  
  MuJoCo 可视化与视频录制。
- `general_motion_retargeting/utils/`  
  各类输入动作数据解析（如 `smpl.py`, `lafan1.py`, `xsens.py`）。
- `scripts/`  
  直接运行的入口脚本（如 `smplx_to_robot.py`, `bvh_to_robot.py`, `xsens_bvh_to_robot.py`）。

---

## 3. 典型运行流程（以离线 SMPL-X 为例）

1. 读取人体动作（`scripts/smplx_to_robot.py` -> `utils/smpl.py`）。
2. 初始化 `GeneralMotionRetargeting(src_human="smplx", tgt_robot=...)`。
3. 每帧将人体关节点设置为 IK 目标并求解。
4. 用 `RobotMotionViewer` 可视化，或保存为机器人动作 `pickle`。

---

## 4. NAO 机器人相关代码概况

### 4.1 入口与映射

NAO 已在全局参数中注册：

- 机器人 XML：`general_motion_retargeting/params.py` 中 `ROBOT_XML_DICT["nao"] -> assets/nao/nao_scene.xml`
- IK 配置：`IK_CONFIG_DICT["smplx"]["nao"] -> general_motion_retargeting/ik_configs/smplx_to_nao.json`
- Viewer 跟随主体：`ROBOT_BASE_DICT["nao"] = "torso"`

同时 `scripts/smplx_to_robot.py` 的 `--robot` 参数已包含 `nao`，可直接调用通用流程。

### 4.2 NAO 的 IK 配置特点

文件：`general_motion_retargeting/ik_configs/smplx_to_nao.json`

关键信息：

- `robot_root_name = "torso"`，`human_root_name = "pelvis"`。
- 使用两阶段任务：`use_ik_match_table1 = true`、`use_ik_match_table2 = true`。
- 在 `human_scale_table` 中对骨盆、脊柱、四肢使用了较小比例（如 pelvis=0.32），以适配 NAO 体型。
- `ik_match_table1/2` 中映射了 NAO 主要
  body：`torso`, `LHip`, `LTibia`, `l_ankle`, `RHip`, `RTibia`, `r_ankle`, `Head`, `LBicep`, `LForeArm`, `RBicep`, `RForeArm`, `l_gripper`, `r_gripper`。

这说明 NAO 重定向目前聚焦在**躯干 + 双腿 + 双臂**的主体运动控制。

### 4.3 NAO 资产与辅助脚本

- 主要模型与资源目录：`assets/nao/`
    - MuJoCo 场景：`assets/nao/nao_scene.xml`
    - URDF：`assets/nao/nao.urdf`
    - 网格：`assets/nao/meshes/V40/*`
    - 纹理：`assets/nao/texture/textureNAO.png`
- 调试脚本：`scripts/vis_nao_frames.py`
    - 功能：打印并可视化 NAO 所有可作为 `frame_type="body"` 的候选 body 坐标系，便于调 IK 映射点。

### 4.4 NAO 当前使用方式（示例）

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/motion.npz
```

如需导出 NAO 动作：

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/motion.npz --save_path /path/to/nao_motion.pkl
```

---

## 5. 维护建议（NAO）

- 新增或微调 NAO 动作效果，优先调整 `smplx_to_nao.json` 的：
    - `human_scale_table`
    - `ik_match_table1/ik_match_table2` 的位置/旋转权重与 offset
- 通过 `scripts/vis_nao_frames.py` 先确认 body 坐标系方向和关节点位置，再改 IK 映射会更稳定。
- 如果要支持 NAO 的其他输入格式（如 BVH/FBX），需在 `params.py` 中补全对应 `IK_CONFIG_DICT` 项并新增配置文件。

---

## 6. 环境安装指引

### 6.1 Python 环境与依赖安装

建议使用 Python 3.10（与项目 `setup.py` 一致）。

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
```

如果遇到渲染相关动态库问题，可补充：

```bash
conda install -c conda-forge libstdcxx-ng -y
```

### 6.2 SMPL-X 依赖与模型文件

GMR 的 SMPL-X 流程依赖 `assets/body_models/smplx/` 下的模型文件，目录结构如下：

```text
assets/body_models/smplx/
  SMPLX_NEUTRAL.pkl
  SMPLX_FEMALE.pkl
  SMPLX_MALE.pkl
```

若使用 SMPL-X 的 pkl 文件，按照 README 说明，需要将已安装 `smplx` 包中的 `body_models.py` 里 `ext` 从 `npz` 调整为 `pkl`。

### 6.3 数据准备（最常见）

- SMPL-X / AMASS：下载后放在任意目录，运行脚本时通过 `--smplx_file` 指向具体 `.npz` 文件。
- OMOMO：先用 `scripts/convert_omomo_to_smplx.py` 转成 SMPL-X 格式再使用。
- BVH（LAFAN1/Nokov/Xsens）：按对应脚本读取（如 `scripts/bvh_to_robot.py`, `scripts/xsens_bvh_to_robot.py`）。

### 6.4 安装后快速自检（NAO）

1. 检查 NAO 模型能否加载并查看候选 body 点：

```bash
python scripts/vis_nao_frames.py
```

2. 用 SMPL-X 数据跑一次 NAO 重定向：

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/motion.npz
```

若以上两步可正常启动 MuJoCo viewer，说明环境和 NAO 链路已基本可用。

### 6.5 数据集下载后常用格式转换命令

> NAO 在当前仓库中使用的是 `src_human="smplx"` 链路，因此最终输入应为 SMPL-X 格式动作文件（通常为 `.npz`）。

#### A. AMASS（已是 SMPL-X）

AMASS 按 README 指引下载的是 SMPL-X 数据，通常可直接用于 NAO 脚本，无需转换：

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/amass_motion.npz
```

#### B. OMOMO -> SMPL-X（项目内转换脚本）

`scripts/convert_omomo_to_smplx.py` 目前是固定路径版本。  
先修改脚本里的这 3 个变量，再执行脚本：

- `motion_path1`
- `motion_path2`
- `target_dir`

执行命令：

```bash
python scripts/convert_omomo_to_smplx.py
```

转换后会在 `target_dir` 下生成逐条动作文件（脚本当前保存为 `.pkl`）。
注意：`scripts/smplx_to_robot.py` 当前通过 `np.load` 读取 `--smplx_file`，实践中建议使用 `.npz` 作为输入；如果你走 OMOMO
脚本链路，需要再整理成兼容的 `.npz` 后再喂给 NAO 主脚本。

#### C. 通用 SMPL -> SMPL-X（命令行可用）

若你手上是 SMPL `.npz`（非 SMPL-X），可用：

```bash
python scripts/smpl_to_smplx.py --src_folder /path/to/smpl_npz --tgt_folder /path/to/smplx_npz --gender neutral
```

单文件转换：

```bash
python scripts/smpl_to_smplx.py --input_file /path/in.npz --output_file /path/out.npz --gender neutral
```

---

## 7. 从下载数据集到运行 NAO 的完整简述流程

1. 准备运行环境  
   创建 `conda` 环境并执行 `pip install -e .`，确保 MuJoCo 相关依赖可用。

2. 准备 SMPL-X body model  
   将 `SMPLX_NEUTRAL.pkl / SMPLX_FEMALE.pkl / SMPLX_MALE.pkl` 放到 `assets/body_models/smplx/`。

3. 准备动作数据（目标：得到可用的 SMPL-X 文件）

- 如果是 AMASS：直接使用下载的 SMPL-X `.npz`。
- 如果是 OMOMO：先跑 `python scripts/convert_omomo_to_smplx.py`（先改脚本内路径），再整理成可被 `--smplx_file`
  直接读取的 `.npz`。
- 如果是普通 SMPL `.npz`：先跑 `python scripts/smpl_to_smplx.py ...` 转成 SMPL-X。

4. （可选）先检查 NAO 模型和可用 body 点

```bash
python scripts/vis_nao_frames.py
```

5. 运行 NAO 重定向主脚本

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/motion.npz
```

6. （可选）导出 NAO 动作结果

```bash
python scripts/smplx_to_robot.py --robot nao --smplx_file /path/to/motion.npz --save_path /path/to/nao_motion.pkl
```

做到第 5 步时，如果 MuJoCo viewer 正常启动且 NAO 有动作，说明“下载数据 -> 转换/准备 -> NAO 运行”链路已打通。

---

## 8. FGD、BC、SRGR 适配到 NAO 的数据转换流程

这一节专门说明：如果我们评估的预测结果不是人体动作 `.npz`，而是 GMR 输出的 `NAO pkl`，那么数据是怎样一步一步被转成各指标可接受的格式的。

先给总览：

```text
BEAT2 原始人体动作 .npz
-> GMR 中间文件 *_amass_compat.npz
-> GMR 输出 NAO 机器人动作 .pkl
-> evaluation/adapters/*.py 做指标适配
-> FGD / BC / SRGR
```

### 8.1 原始输入和中间文件分别是什么

#### A. BEAT2 原始 `.npz`

典型来自：

```text
datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/*.npz
```

里面通常有：

- `poses`
- `trans`
- `betas`
- `gender`
- `mocap_frame_rate`

其中：

- `poses` 是人体 SMPL-X 轴角参数，通常一帧里包含 root + body + 其他关节
- `trans` 是人体根节点平移

#### B. GMR 中间 `.npz`

由 `scripts/beat2_to_robot.py` 生成：

```text
motion_data/BEAT2/converted/*_amass_compat.npz
```

这个文件会把原始 `BEAT2 npz` 整理成 GMR 和 `smplx_to_robot.py` 方便读取的结构：

- `pose_body`
- `root_orient`
- `trans`
- `betas`
- `gender`
- `mocap_frame_rate`

可以理解成：

- 把原始 `poses` 拆成“根旋转 + 身体旋转”
- 保留位移、体型、性别和帧率

#### C. GMR 输出的 NAO `.pkl`

由 `scripts/smplx_to_robot.py` 或 `scripts/beat2_to_robot.py` 最终生成：

```text
motion_data/BEAT2/retargeted/*_nao.pkl
```

里面核心字段是：

- `fps`
- `root_pos`
- `root_rot`
- `dof_pos`

含义：

- `root_pos`：机器人根节点平移
- `root_rot`：机器人根节点旋转，保存时是 `xyzw`
- `dof_pos`：机器人所有可动关节的自由度值

这已经不再是人体 SMPL-X 数据，而是 **NAO 机器人在 MuJoCo 中的运动状态序列**。

---

### 8.2 FGD 适配到 NAO pkl 的步骤

对应代码：

- `evaluation/fgd.py`
- `evaluation/adapters/pkl_motion.py`
- `evaluation/adapters/beat2.py`

#### FGD 的目标输入

当前工程里，FGD 最终吃的是：

```text
(T, 55, 6)
```

也就是：

- `T` 帧
- 55 个“类 SMPL-X 关节槽位”
- 每个槽位一个 `rotation 6D`

#### GT（人体真值）怎么转

如果 GT 是 `BEAT2/GMR npz`：

1. 读取 `.npz`
2. 整理成 `55 * 3 = 165` 维 axis-angle
3. reshape 成 `(T, 55, 3)`
4. 再转成 `(T, 55, 6)` 的 rot6d

实现位置：

- `evaluation/adapters/beat2.py`
- `beat2_to_fgd_rot6d()`

#### Pred（NAO pkl）怎么转

如果 Pred 是 `NAO pkl`：

1. 读取 `root_pos / root_rot / dof_pos`
2. 把 `root_rot` 从 `xyzw` 转成旋转向量 `rotvec`
3. 拼成一个状态向量：

```text
[root_pos(3), root_rotvec(3), dof_pos(D)]
```

4. 把这个状态向量截断或补零到 `165` 维
5. reshape 成 `(T, 55, 3)`，当作“临时代理 axis-angle”
6. 再转成 `(T, 55, 6)` rot6d

实现位置：

- `evaluation/adapters/pkl_motion.py`
- `pkl_to_axis_angle_proxy()`
- `pkl_to_fgd_rot6d_proxy()`

#### 这一条链路的本质

FGD 对 NAO pkl 的适配目前不是“真实恢复每个 SMPL-X 关节旋转”，而是：

```text
机器人状态向量
-> 代理人体 axis-angle
-> rot6d
-> FGD encoder
```

因此它是 **临时工程代理表示**。

需要明确：

- 这条映射的目的，是让 `NAO pkl` 能先接入现有 `FGD` evaluator，方便联调
- 它目前**没有明确的论文定义或标准工程实现支撑**
- 它解决的是“输入 shape 兼容”，不是“动作语义严格正确”

所以当前 `FGD` 的 `NAO pkl` 适配只能视为：

- 可运行的临时方案
- 不建议直接作为正式论文结论依据

如果后续要把 `FGD` 变成正式评测模块，更合理的方向是：

1. `机器人 -> 统一关键点位置空间 -> 重新定义/适配 FGD 输入`
2. `机器人 -> 真实关节旋转空间 -> 明确 joint layout 映射 -> 再决定是否能进入 FGD encoder`

---

### 8.3 BC 适配到 NAO pkl 的步骤

对应代码：

- `evaluation/bc.py`
- `evaluation/adapters/pkl_motion.py`
- `evaluation/adapters/beat2.py`

#### BC 的目标输入

当前接入的 `emage_evaltools.BC` 最终吃的是一个二维数组：

```text
(T, 55*3)
```

代码内部会再 reshape 成：

```text
(T, J, 3)
```

然后按关节速度去找 motion beat，再与音频 onset 对齐。

#### GT（人体真值）怎么转

1. 读取 GT `.npz`，兼容两种来源：
    - 原始 `BEAT2 smplxflame_30/*.npz`
    - `motion_data/BEAT2/converted/*_amass_compat.npz`
2. 转成 `(T, 55, 3)` 的 axis-angle
3. flatten 成 `(T, 165)`

实现位置：

- `evaluation/adapters/beat2.py`
- `beat2_to_axis_angle()`
- `flatten_axis_angle_sample()`

#### Pred（NAO pkl）怎么转

1. 读取 `root_pos / root_rot / dof_pos`
2. `root_rot(xyzw)` -> `rotvec`
3. 拼成状态向量 `[root_pos, root_rotvec, dof_pos]`
4. 截断或补零到 `165` 维
5. reshape 成 `(T, 55, 3)` 的临时代理 axis-angle
6. flatten 成 `(T, 165)`

实现位置：

- `evaluation/adapters/pkl_motion.py`
- `pkl_to_axis_angle_proxy()`
- `flatten_axis_angle_proxy()`

#### 这一条链路的本质

BC 当前也不是直接从 NAO 的真实身体关键点位置来算，而是沿用了一条代理通道：

```text
NAO pkl
-> 代理 axis-angle 序列
-> flatten
-> BC 节拍一致性计算
```

所以当前 BC 的意义更接近：

- “机器人状态变化节奏是否和音频节拍一致”
- 而不是严格意义上的“人体手势关键点节拍一致性”

这里也需要明确：

- 当前 `BC` 的 `NAO pkl` 适配方式与 `FGD` 一样，属于**临时代理方案**
- 它主要用于工程验证和流程打通
- 它同样**没有明确的论文定义或标准工程实现支撑**

因此：

- 当前 `BC` 结果可以作为辅助参考
- 不建议直接把这条代理方案作为论文中的正式指标实现

---

### 8.4 SRGR 适配到 NAO pkl 的步骤

对应代码：

- `evaluation/srgr.py`
- `evaluation/emage_evaltools/mertic.py` 中的 `SRGR`

SRGR 和 FGD/BC 的处理思路不同。

FGD/BC 当前走的是“代理人体表示”路线。  
SRGR 这里采用的是更直观的 **共享位置空间**：

```text
GT 人体动作
-> GMR 对齐后的人体目标点位置

Pred 机器人动作 pkl
-> MuJoCo 正向运动学得到 NAO body 位置

两边都落到同一组 GMR 匹配点上
-> SRGR
```

#### GT（人体真值）怎么转

1. 读取 `BEAT2/GMR npz`
2. 用 `smplx` body model 还原每一帧人体关节位置与旋转
3. 初始化：

```text
GeneralMotionRetargeting(src_human="smplx", tgt_robot="nao")
```

4. 不做机器人 IK 求解，只复用它的前半段对齐逻辑：
    - 人体尺度缩放
    - root 对齐
    - offset 修正
5. 取 `ik_match_table1`（默认）中 **位置权重大于 0** 的那组匹配点
6. 形成：

```text
(T, J, 3)
```

这里的 `J` 不是固定 55，而是“NAO 配置里真正拿来做位置约束的匹配点数”。

这一步得到的是：

**“经过 GMR 规则缩放/偏移后的目标人体点位”**

#### Pred（NAO pkl）怎么转

1. 读取 `NAO pkl`
2. 把每一帧的：
    - `root_pos`
    - `root_rot(xyzw -> wxyz)`
    - `dof_pos`
      组装成 MuJoCo 的 `qpos`
3. 加载 NAO 的 MuJoCo XML：

```text
assets/nao/nao_scene.xml
```

4. 对每一帧做一次 `mj_forward`
5. 读取和 `ik_match_table1` 对应的机器人 body 的世界坐标 `xpos`
6. 形成：

```text
(T, J, 3)
```

这一步得到的是：

**“NAO 在仿真中这些关键 body 的真实位置”**

#### semantic 权重怎么进入 SRGR

`SRGR.run(results, targets, semantic=...)` 支持每帧一个 semantic 权重。

当前实现支持：

- 不传 semantic：默认全 1
- 传 `.npy`
- 传 `.csv`
- 传 `.txt`

批量模式下可以通过 `--semantic_dir` 按 GT 相对路径匹配对应语义文件。

注意：

- `SRGR` 当前**不读取音频**
- 和 `BC` 不同，它的附加输入是 `semantic` 权重，而不是 `audio onset`

#### SRGR 当前的工程含义

SRGR 在这个项目里的适配，不再是“把机器人状态硬塞进人体姿态槽位”，而是：

```text
比较 NAO 在关键 body 上的位置
是否足够接近 GMR 期望它追踪的人体目标点
再用 semantic 权重做逐帧加权
```

这个适配方式比 FGD/BC 目前的代理方案更接近“机器人是否真的把该做的动作做到了”。

---

### 8.5 三个指标的区别（站在当前工程实现角度）

- `FGD`
    - 输入：临时代理 rot6d
    - 关注：整体动作分布是否接近
    - 当前 NAO 适配方式：状态向量代理
    - 备注：当前方案无论文/标准工程实现背书，只适合联调和辅助分析

- `BC`
    - 输入：临时代理 `(T, 55*3)` 序列 + 音频
    - 关注：动作节拍是否和音频 onset 对齐
    - 当前 NAO 适配方式：状态向量代理
    - 备注：当前方案无论文/标准工程实现背书，只适合联调和辅助分析

- `SRGR`
    - 输入：共享关键点位置 `(T, J, 3)` + semantic 权重
    - 关注：关键动作点是否真正完成，并考虑语义权重
    - 当前 NAO 适配方式：GMR 对齐目标点 vs MuJoCo 真实 body 点

---

### 8.6 当前建议的实际使用顺序

如果是 NAO 机器人评估，建议先这样用：

1. `SRGR`
    - 因为它最接近“NAO 有没有把该追踪的关键点做到位”

2. `BC`
    - 看节拍是否和音频对齐

3. `FGD`
    - 作为整体动作分布的参考分数

原因是：

- `SRGR` 当前适配最贴近机器人空间
- `FGD/BC` 目前仍带代理表示成分，更适合作为辅助指标而不是唯一指标

如果后续要把 `FGD/BC` 升级成正式评测模块，建议优先推进：

1. 设计一套明确的 joint layout 映射  
   目标是说明“NAO 的哪些关节 / body 对应到统一评估骨架中的哪些关节”

2. 明确统一的动作表示  
   可选：
   - 关键点位置空间
   - 真实局部关节旋转空间

3. 在这个统一空间上重新定义 `FGD/BC` 的输入  
   只有这样，指标数值才会有更强的可解释性，也更适合作为论文中的正式方法组成部分
