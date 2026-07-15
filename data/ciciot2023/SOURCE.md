# CICIoT2023 (dev sample) — source & provenance

CIC/UNB CIC IoT Dataset 2023 (33 attacks, 7 categories, 105-device IoT testbed).
Official: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>. Dev sample
(pre-split, downsampled ~1.34M rows) from mirror
<https://huggingface.co/datasets/lacg030175/CIC-IoT-2023> (`random/` split).
Full 2.1GB parquet: `lacg030175/CIC-IoT-2023-full`; raw 18GB CSVs: `bencorn/CIC-IoT-2023`.
3 label levels: `label` (binary), `attack_class` (8-category), `Label` (fine, ~34).
Cite: Neto et al., *CICIoT2023*, 2023.

| File | Rows | Cols | SHA-256 |
| --- | ---: | ---: | --- |
| `train.parquet` | 1,073,851 | 42 | `4a9c585c6c1c7ef4cbaea624e71d0d960736ed8ed40006dc4bd918abfa09cc1d` |
| `test.parquet` | 268,463 | 42 | `dbbaa3a7842d193cd04b6e5eba87e04b5d75b8951a62029dc29f80ae0e358ac2` |
