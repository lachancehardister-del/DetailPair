# DetailPair project instructions

User authorization recorded on 2026-10-13 (Asia/Shanghai): automatically synchronize future project code, data, weights and experiment results to this GitHub repository, and keep logs.

- After requested project work, commit and push the relevant code, provided project data, weights, experiment results and a truthful work log. This user authorization persists; do not ask again merely to commit or push.
- Preserve original data, old versions and failed runs. Do not force-push or discard unrelated changes. Never include credentials, virtual environments, machine settings or caches.
- Label synthetic checks as synthetic. Separate actual validation from unrun Windows/CUDA/real-data checks; never infer performance gains from plumbing tests.
- Record changes, artifact provenance/hashes, checks and limitations in WORK_LOG.md; record the remote commit and synchronization outcome after verifying remote state.
- Use Git LFS for large data or weights when supported; do not silently omit artifacts that cannot be uploaded. Report exact blockers.
- Use this existing isolated checkout; do not create a Git worktree unless requested.
- The v0.2.5 task document is background until the user asks for implementation. Do not launch long training or change experiment protocols merely because attached documents mention them.
