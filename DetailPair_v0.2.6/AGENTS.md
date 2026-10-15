# DetailPair project instructions

User authorization recorded on 2026-10-13 (Asia/Shanghai): automatically synchronize future project code, data, weights and experiment results to this GitHub repository, and keep logs.

- After requested project work, commit and push the relevant code, provided project data, weights, experiment results and a truthful work log. This user authorization persists; do not ask again merely to commit or push.
- Preserve original data, old versions and failed runs. Do not force-push or discard unrelated changes. Never include credentials, virtual environments, machine settings or caches.
- Label synthetic checks as synthetic. Separate actual validation from unrun Windows/CUDA/real-data checks; never infer performance gains from plumbing tests.
- Record changes, artifact provenance/hashes, checks and limitations in this version’s WORK_LOG.md; record the remote commit and synchronization outcome after verifying remote state.
- Use Git LFS for large data or weights when supported; do not silently omit artifacts that cannot be uploaded. Report exact blockers.
- Use this existing isolated checkout; do not create a Git worktree unless requested.
- The v0.2.5 task document is background until the user asks for implementation. Do not launch long training or change experiment protocols merely because attached documents mention them.

## Work / Codex / local iteration

Work reviews local experiment results and provides a research plan. Codex implements the user-approved plan, runs available engineering checks, writes logs, synchronizes changes, and delivers a full Windows-ready code ZIP. The user runs locally and returns the results to Work. Do not initiate long training, extend budgets or research scope without the approved plan.

The 2026-10-13 reuse-evaluation release is DetailPair_v0.2.5. Its approved plan keeps real trained weights/data local and excludes them from return ZIPs; synchronize supplied review materials and reports, not unavailable or explicitly excluded real weights. Preserve v0.2.4. CPU random-checkpoint checks are synthetic and do not establish real detection performance.

## Version directory layout

Keep all visible root-level files limited to README.md. Each new version gets a new DetailPair_vX.Y.Z directory. Keep its code, configs, inputs, reports, logs, tools and release packages within that version; preserve previous versions. Put project instructions inside each version directory, and link versions from the root introduction. Hidden Git configuration remains at repository root.

## Continuing authorization (2026-10-15 Asia/Shanghai)

When the user supplies a Work plan and result package in this iteration, proceed directly to inspection, implementation, verification, Git synchronization and a complete versioned Windows code ZIP. Do not ask the same execution confirmation again. Use GPU for local real evaluation, with no CPU fallback. Version0.2.6 is a bounded timing/clean-diagnostic task: no training, corruption sweep, YOLO11s or final test. Preserve the submitted QQ-blocked and drifting runs. Root remains introduction only; all new materials live in this version directory.
