# CICIDS2017 — source & provenance

CIC/UNB, <https://www.unb.ca/cic/datasets/ids-2017.html>. CICFlowMeter flow features (5 days of traffic, 8 CSVs). Mirror <https://huggingface.co/datasets/c01dsnap/CIC-IDS2017>. No official train/test split — we construct a stratified one in preprocessing. Known issues (whitespace in headers, NaN/Inf, class imbalance) are cleaned downstream. Cite: Sharafaldin et al., ICISSP 2018.

| File | Rows | Cols | SHA-256 |
| --- | ---: | ---: | --- |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 79 | `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34` |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 79 | `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 79 | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 79 | `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 79 | `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 79 | `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 79 | `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b` |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 79 | `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee` |
