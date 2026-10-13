# DetailPair cloud development

Use the existing isolated checkout at /workspace/DetailPair. Do not create a Git worktree unless explicitly requested. The imported experiment package is /workspace/DetailPair/DetailPair_v0.2.4. Preserve its source files and the original uploaded status/task documents. v0.2.5 implementation and long real-data training require a separate user request.

Activate the prepared CPU environment:

```bash
source /workspace/venvs/detailpair/bin/activate
export PIP_CACHE_DIR=/workspace/.cache/pip
export YOLO_CONFIG_DIR=/workspace/DetailPair/DetailPair_v0.2.4/local_settings
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd /workspace/DetailPair/DetailPair_v0.2.4
python -m pip check
python tests/check_invariants.py
python tests/check_novelty_controls.py
python run.py smoke
```

No background services are needed. For an explicit short synthetic end-to-end readiness check, run `python run.py synthetic --device cpu --variant B6`; it runs one synthetic epoch and validates checkpoint save/reload, validation and prediction. Confirm exit status zero and the new runs/synthetic_B6_*/completed.json and status.json both say completed/success. These results are plumbing checks, not defect detection performance evidence. `python run.py pack` preserves the supported result packaging workflow, excluding source images/labels and weights by design.

This cloud instance has no visible GPU, NEU-DET data, trained weights or historical result ZIPs. Do not automatically launch compare_long.py, fetch/re-split real data, run final test evaluation or claim CUDA/Windows/real-time performance validation. Read onboarding/REVIEW.md and the uploaded status (latest entries take precedence) for evidence boundaries. No web server or preview is required.
