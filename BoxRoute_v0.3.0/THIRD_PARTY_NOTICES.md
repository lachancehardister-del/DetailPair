# Third-party sources and adaptations

Ultralytics is fixed at8.3.221/e544db43838fd058370cad216bf82a7639f29b35. See sources/LICENSE_ultralytics.txt (AGPL-3.0) and the fixed source copies. Native model/head/BCE-CIoU-DFL/TAL/metric implementation is retained; the shared square data pipeline and post-layer16 additions are explicitly different from the official default training command.

PKINet-v2 PKSModule is from a33aa22d188c9946cc83fba60e3bb8ac0ec82ff7, provided as sources/pkinet_v2_fixed.py, licensed under sources/LICENSE_pkinet.txt (Apache-2.0). boxroute/pks.py copies only the reviewed PKSModule and its BN fusion helper, replacing default BN construction with BatchNorm2d. Original five branches, max19 receptive scope, conv0/conv1 and deploy conversion remain. YOLO P3 placement and outer trainable gamma residual are project adaptations. This is not full PKINet-v2 reproduction.

Strip/LSK/MPSL source excerpts and original licenses supplied by Work remain research references in sources; they are not additional runtime models or completed replications. SOURCE_RECORDS.json, SOURCE_RETRIEVAL_LOG.json, SEARCH_LOG.json, BASELINE_AND_NEAREST_AUDIT.md and NOTICE_SOURCES.md preserve attribution and coverage limitations.

The public official COCO initialization is a supplied data resource (not user-trained unpublished weights): sources/official_yolo11n_v830.pt, SHA256 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1,5613764 bytes. URL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt . Work downloaded/hashed it; Codex has now loaded it in PyTorch engineering checks. User images/labels/trained weights remain local and are excluded from result archives.

legacy_detailpair/model.py is a byte-identical copy of the archived project class for B0/native equivalence checks only. No archived paired/Haar training is executed by this new suite.
