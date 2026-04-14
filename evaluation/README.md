# 评估模块说明（FGD + BC + SRGR）

本目录用于在 GMR 项目中统一管理评估逻辑，当前已完成：

- `FGD`：基于 `emage_evaltools` 的 Frechet Gesture Distance 计算。
- `BC`：基于 `emage_evaltools` 的 Beat Consistency 计算。
- `SRGR`：基于 `emage_evaltools` 的 Semantic Relevance Gesture Recall 计算。
- 输入严格按你当前转换链路处理：允许 `root + body` 后对其余关节补零，不做严格 FGD 全量约束。

## 本次修改概况（pred_format）

本次新增了 `pred_format` 接口，默认值为 `npz`，可选 `pkl`，其余已有参数保持不变：

- `scripts/run_fgd.py` 新增 `--pred_format {npz,pkl}`
- `scripts/run_bc.py` 新增 `--pred_format {npz,pkl}`
- 默认不传时仍是 `npz`，与原先行为一致

当 `pred_format=pkl` 时，系统会把机器人 retarget 的 `pkl`（`root_pos/root_rot/dof_pos`）转换为可评估的 proxy 表示：

- FGD：转换为 `(T,55,6)` 的 proxy rot6d 后进入 FGD
- BC：转换为 `(T,55,3)` 的 proxy 轴角后计算节拍一致性

说明：这属于工程可用的代理转换，和直接使用人体真值表示相比会引入额外误差。

## 目录结构

- `evaluation/adapters/beat2.py`
    - 读取 BEAT2 / GMR 兼容 `npz`
    - 转换 `FGD` 需要的 `rot6d`
    - 转换 `BC` 需要的轴角序列（当前链路兼容版）
- `evaluation/fgd.py`
    - FGD 封装、批量配对、资源检查
- `evaluation/bc.py`
    - BC 封装、单文件/批量计算辅助
- `evaluation/srgr.py`
    - SRGR 封装、GT/Pred 对齐、semantic 权重读取
- `evaluation/emage_evaltools/`
    - 外部实现代码（已放到 `evaluation` 下）
- `scripts/run_fgd.py`
    - FGD 命令行入口
- `scripts/run_bc.py`
    - BC 命令行入口
- `scripts/run_srgr.py`
    - SRGR 命令行入口

## 权重与模型文件放置位置

统一放到：

- `evaluation/weights/emage/AESKConv_240_100.bin`
- `evaluation/weights/emage/mean_vel_smplxflame_30.npy`
- `evaluation/weights/emage/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`

说明：`evaluation/emage_evaltools` 里同名大文件通常是 Git LFS 指针，不可直接用于计算。请按下面命令下载真实文件。

## 权重下载命令

在项目根目录执行：

```bash
python scripts/download_hf.py \
  --repo-id H-Liu1997/emage_evaltools \
  --repo-type model \
  --local-dir evaluation/weights/emage \
  --file-map "AESKConv_240_100.bin::AESKConv_240_100.bin" \
  --file-map "mean_vel_smplxflame_30.npy::mean_vel_smplxflame_30.npy" \
  --file-map "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz::smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
```

如果环境还没有 `huggingface_hub`，先安装：

```bash
pip install huggingface_hub
```

## 依赖建议

建议在云端环境确认以下包可用：

```bash
pip install numpy scipy torch librosa matplotlib wget
```

## FGD 运行命令

1. 单文件评估

```bash
python scripts/run_fgd.py \
  --gt /path/to/gt_motion.npz \
  --pred /path/to/pred_motion.npz \
  --weights_root evaluation/weights/emage \
  --device cuda
```

2. 目录批量评估（按相对路径配对 `gt` 与 `pred`）

```bash
python scripts/run_fgd.py \
  --gt /path/to/gt_dir \
  --pred /path/to/pred_dir \
  --weights_root evaluation/weights/emage \
  --device cuda
```

3. `pred` 使用 retarget 后的 `pkl`

```bash
python scripts/run_fgd.py \
  --gt /path/to/gt_motion.npz \
  --pred /path/to/pred_motion.pkl \
  --pred_format pkl \
  --weights_root evaluation/weights/emage \
  --device cuda
```

4. 如需严格 FGD 全量姿态（默认关闭）

```bash
python scripts/run_fgd.py ... --strict_fgd
```

默认模式不启用 `--strict_fgd`，会按当前 GMR 转换链路允许补零关节。

## BC 运行命令

1. 仅评估 `pred`（单文件 + 单音频）

```bash
python scripts/run_bc.py \
  --pred /path/to/pred_motion.npz \
  --audio /path/to/audio.wav \
  --weights_root evaluation/weights/emage
```

2. 评估 `pred` 与 `gt`（单文件）

```bash
python scripts/run_bc.py \
  --pred /path/to/pred_motion.npz \
  --gt /path/to/gt_motion.npz \
  --audio /path/to/audio.wav \
  --weights_root evaluation/weights/emage
```

3. 目录批量（同一音频）

```bash
python scripts/run_bc.py \
  --pred /path/to/pred_dir \
  --gt /path/to/gt_dir \
  --audio /path/to/audio.wav \
  --weights_root evaluation/weights/emage
```

4. 目录批量（按相对路径匹配音频）

`audio_dir` 下音频路径规则：`xxx/yyy.npz -> xxx/yyy.wav`

```bash
python scripts/run_bc.py \
  --pred /path/to/pred_dir \
  --gt /path/to/gt_dir \
  --audio_dir /path/to/audio_dir \
  --weights_root evaluation/weights/emage
```

5. `pred` 使用 retarget 后的 `pkl`

```bash
python scripts/run_bc.py \
  --pred /path/to/pred_motion.pkl \
  --gt /path/to/gt_motion.npz \
  --pred_format pkl \
  --audio /path/to/audio.wav \
  --weights_root evaluation/weights/emage
```

## 输出说明

两个脚本都会输出 JSON，便于你直接贴回对话里做下一步排查。

- `run_fgd.py` 输出 `fgd`
- `run_bc.py` 输出 `pred_bc`、`gt_bc`、`bc_gap_pred_minus_gt`
- `run_srgr.py` 输出 `srgr`

## SRGR 运行命令

1. 单文件评估（Pred 为 NAO pkl，GT 可为原始 BEAT2 `.npz` 或 GMR 的 `*_amass_compat.npz`）

```bash
python scripts/run_srgr.py \
  --gt /path/to/gt_motion.npz \
  --pred /path/to/pred_motion.pkl \
  --pred_format pkl \
  --robot nao
```

2. 单文件评估并提供 semantic 权重

```bash
python scripts/run_srgr.py \
  --gt /path/to/gt_motion.npz \
  --pred /path/to/pred_motion.pkl \
  --pred_format pkl \
  --robot nao \
  --semantic /path/to/semantic.csv
```

3. 目录批量评估

```bash
python scripts/run_srgr.py \
  --gt /path/to/gt_dir \
  --pred /path/to/pred_dir \
  --pred_format pkl \
  --robot nao
```

4. 目录批量评估并按相对路径匹配 semantic 文件

```bash
python scripts/run_srgr.py \
  --gt /path/to/gt_dir \
  --pred /path/to/pred_dir \
  --pred_format pkl \
  --robot nao \
  --semantic_dir /path/to/semantic_dir
```

说明：

- `SRGR` 不像 `BC` 那样依赖音频，不需要 `--audio`
- `semantic` 是可选输入；不提供时默认按全 1 权重计算
- `semantic` 是权重，目前暂时不考虑计算进去

## 备注

- 本地这次只做代码落地，不做依赖环境运行自检。
- 若云端运行报错，请把完整报错和命令贴回，我会继续对齐修复。
