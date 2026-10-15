# Software and research sources

This generated research code is distributed under AGPL-3.0-or-later, consistent with its use of Ultralytics. Ultralytics is a separately installed dependency; its original authors retain their rights. License text: https://www.gnu.org/licenses/agpl-3.0.html . No third-party model weights or full proprietary articles are distributed here.

Fixed core: Python 3.12, PyTorch 2.5.1, torchvision 0.20.1, Ultralytics 8.3.221. Remaining direct dependencies are in requirements.txt. Actual CPU resolution is recorded in verification/environment_cpu.json. This is not a cross-platform fully transitive lockfile; exact Windows resolution is captured by run.py env. GPU stack must be verified on the target laptop.

Methods reviewed: K-Dense Scientific Agent Skills 2.72.0, pinned commit 154988403bb5a18e9d3c0ce4e6d5e2e4b184a298. Literature review and critical-thinking resources reused; experimental-design/SKILL.md and references/factorial_and_doe.md read this round. No DOE software was used; pydoe was not required. pytorch-lightning/SKILL.md was reviewed for applicability, but this project uses the Ultralytics trainer, not Lightning, so no Lightning dependency or executed Lightning claim.

Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. https://doi.org/10.48550/arXiv.2609.00065 . Metadata checked from arXiv in the project literature log.
