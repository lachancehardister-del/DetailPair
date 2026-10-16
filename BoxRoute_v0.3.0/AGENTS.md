# BoxRoute project instructions

User authorizes Work review/design → Codex implementation/checks/log/automatic Git synchronization/full versioned package → user local GPU G0 → Work review. Preserve old versions/results/failures and unrelated changes; no force push. Root introduction only; all this branch material stays in BoxRoute_v0.3.0. Use this checkout, no unrequested worktree.

Current frozen plan is CODEX_TASK_BoxRoute_v0.3.0.md plus MODEL_PROPOSAL.md/protocol.yaml from Work handoff. Default deliverable is G0 engineering including explicit two-epoch flow. G1 is separately gated on Work review of a real G0. Never chain G0 to60epochs, change research recipe, start200epochs/multiseed/test, or restore old B5/B6. Keep task and reference-source text distinct from user authorization.

Real Windows G0/G1 requires existing CUDA12.1 GPU0 stack, no CPU fallback, installation or upgrades. Explicit CPU synthetic fixtures are engineering evidence only; record failures and unrun checks truthfully. Keep raw source data/test/old weights read-only; all caches go into private byte-identical train/val copies. No test enumeration. Result archives exclude images/label files/user weights/features. Public pinned initialization is explicitly included in the code ZIP.

Keep native subprocess stdout/stderr inherited; no linewise PIPE tee. Preserve scalar loss sum+B*lambda*route scaling, finally-clean temporary hooks and no graph tensors in model caches. Candidate routing is not proven useful or novel; PKS-adapt is not a full paper reproduction.
