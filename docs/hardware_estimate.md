# Hardware and storage estimate

Collected 2026-07-23. The repository filesystem reported 804 GB free and `/data2` 1.6 TB free, so the 250 GB download gate passes. The existing official train/validation tar is 26,290,307,072 bytes and was extracted to `/data2/2025/ldh/SatVideoIRSDT_v1_train_val/` at about 25G. File counts are 199,600 under train and 46,175 under val. Directory counts are 1,178 train and 255 val; val also contains the root metadata file `geo_val_added.txt`, so this filesystem observation is kept separate from V3's official 1000/200/200 and the older 1001/202/200 figures.

Measured: Python 3.9.20, Linux x86_64, two-socket Intel Xeon Platinum 8280 host with 112 logical/56 physical cores and 251 GiB RAM (206 GiB available at collection), eight NVIDIA GeForce RTX 4090 devices with 24,564 MiB each, driver 535.183.01. CUDA/PyTorch runtime was not imported.

Planning minimum for this no-training round: 8 CPU cores, 32 GB RAM, 250 GB free NVMe; GPU optional. Recommended later development: 16 CPU cores, 64 GB RAM, one 24 GB NVIDIA GPU and 500 GB NVMe. Two 24 GB GPUs are an experiment-parallelism extension, not a baseline requirement.

These are planning recommendations, not benchmark measurements. A storage estimate should use `compressed tar + extracted bytes + prediction zip + temporary working margin`; the extracted-byte term must be measured after access is granted. Keep at least 20% free space and avoid caching all video frames in RAM. No training, cache build, checkpoint, or experiment log was created in this round.
