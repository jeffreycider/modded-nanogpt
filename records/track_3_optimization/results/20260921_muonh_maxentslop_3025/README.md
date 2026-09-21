# MaxEntSlop momentum on MuonH fast-slow decay, 3025 steps

Eight consecutive seeds (0–7) trained on the 3025-step schedule with `train_gpt_muonh_maxentslop.py` and `--min_lr 0.000436` (the MuonH learning rate ends at that value instead of zero; see the PR). The script also removes, before Newton–Schulz, the part of the summed gradients older than 63 steps that is parallel to the current weight matrix (`OLD_HISTORY_PROJECTION_AGE` in the script). Per-seed final validation losses in `summary.tsv`, full file logs (embedded source and environment) in `seed0.txt … seed7.txt`. Hardware: Lambda 8×A100-80GB, PyTorch 2.11.0+cu128. `momentum_kernels.gif` shows the momentum kernels over training; `val_loss_8seed.png` the validation loss of the eight seeds against the K-Maxwell record (PR #359).

```bash
for seed in 0 1 2 3 4 5 6 7; do
  torchrun --standalone --nproc_per_node=8 train_gpt_muonh_maxentslop.py --seed "$seed" --train_steps 3025 --min_lr 0.000436 || exit
done
```
