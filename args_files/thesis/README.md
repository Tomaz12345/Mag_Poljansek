# Thesis Experiment Configurations

The 61 configurations behind the results table in the thesis and in the main
[README](../../README.md#results). Each file is a complete `args_file` for
`src.train`, `src.predict` and `src.continuous_run`.

Resolved with the code in this repository, every file gives exactly the settings recorded in
the original run's `config/config.json`. Only the directory paths were changed (to `data/` and
`logs/`), and `save_freq` was set to the value the run actually used.

## Where each table column comes from

| Column | Architectures | Foundation models |
|--------|---------------|-------------------|
| Baseline | `models_baseline/<model>_baseline.txt` (ViT: `models_default/vit_default.txt`) | – |
| Pretrained | `models_pretrain/<model>_pretrain.txt` | FlexiMo: `models_pretrain/fleximo_pretrain.txt` |
| Frozen | `models_freeze_encoder/<model>_freeze_encoder.txt` | `models_freeze_encoder/<fm>_freeze_encoder.txt` |
| GCE | `models_gce/<model>_gce.txt` | `models_gce/<fm>_freeze_encoder_gce.txt`, `fleximo_pretrain_gce.txt` |
| SCE | `models_sce/<model>_sce.txt` | `models_sce/<fm>_freeze_encoder_sce.txt`, `fleximo_pretrain_sce.txt` |
| Edges | `models_border_exclusion/<model>_border_exclusion.txt` | – |
| Edges + Pre. | `models_border_exclusion_pretrain/<model>_border_exclusion_pretrain.txt` | `<fm>_freeze_encoder_border_exclusion.txt`, `fleximo_pretrain_border_exclusion.txt` |

`<model>` is one of `unet`, `deeplabv3plus`, `swin`, `unetformer`, `oneformer`, `vit`;
`<fm>` is one of `dinov2`, `croma`, `copernicus_fm`, `deo_fm`. There is no configuration for
ViT in the Edges column.

## Running

Place SEN12MS and DFC2020 under `data/` (layout in the main README's [Data](../../README.md#data)
section), then for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.continuous_run \
  --args_file_path args_files/thesis/models_pretrain/unet_pretrain.txt
```

The original runs used one NVIDIA A100 40 GB GPU each.
