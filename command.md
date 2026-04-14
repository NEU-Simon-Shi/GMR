# Commands

ps: 3D_GMR 是 GMR 的机器人数据文件通过 mujoco 转换而成的 (T, J, 3) 三维坐标数据

## BC

### 1. Pred only: GMR / 3D_GMR

```bash
python scripts/run_bc.py \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz \
  --pred_format gmr \
  --audio datasets/BEAT2/beat_english_v2.0.0/wave16k/3_solomon_0_6_6.wav
```

### 2. GT + Pred: BEAT2 vs GMR / 3D_GMR

```bash
python scripts/run_bc.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/3_solomon_0_6_6.npz \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz \
  --pred_format gmr \
  --audio datasets/BEAT2/beat_english_v2.0.0/wave16k/3_solomon_0_6_6.wav
```

### 3. Batch pairs: BEAT2 vs GMR / 3D_GMR

```bash
python scripts/run_bc.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --pred motion_data/BEAT2/3D_GMR \
  --pred_format gmr \
  --audio_dir datasets/BEAT2/beat_english_v2.0.0/wave16k
```

## SRGR

### 1. Single pair

```bash
python scripts/run_srgr.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/3_solomon_0_6_6.npz \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz
```

### 2. Single pair with semantic file

```bash
python scripts/run_srgr.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/3_solomon_0_6_6.npz \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz \
  --semantic /path/to/semantic.csv
```

### 3. Batch pairs

```bash
python scripts/run_srgr.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --pred motion_data/BEAT2/3D_GMR
```

### 4. Batch pairs with semantic dir

```bash
python scripts/run_srgr.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --pred motion_data/BEAT2/3D_GMR \
  --semantic_dir /path/to/semantic_dir
```

## Smoothness

### 1. Pred only

```bash
python scripts/run_smoothness.py \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz
```

### 2. GT + Pred

```bash
python scripts/run_smoothness.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/3_solomon_0_6_6.npz \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz
```

### 3. GT + Pred with dt normalization

```bash
python scripts/run_smoothness.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/3_solomon_0_6_6.npz \
  --pred motion_data/BEAT2/3D_GMR/3_solomon_0_6_6_nao.npz \
  --use_dt_normalization
```

### 4. Batch pairs

```bash
python scripts/run_smoothness.py \
  --gt datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --pred motion_data/BEAT2/3D_GMR
```

### 5. Pred only batch

```bash
python scripts/run_smoothness.py \
  --pred motion_data/BEAT2/3D_GMR
```
