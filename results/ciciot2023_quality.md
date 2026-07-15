# CICIoT2023 (parquet dev) — data-quality & leakage audit

All features numeric → scale-only preprocessing. This dev sample uses a **random** split (in-distribution — caveat every score). No IP/port/timestamp columns → host/time leakage already avoided.

### train  (1,073,851 rows x 39 features)

- missing cells: **0**
- infinite cells: **0**
- duplicated rows: **84,370** (7.86%)
- constant features: **0** 
- near-constant (>99.9% one value): **2** ['Telnet', 'SMTP']
- binary/category consistency: **OK**

### test  (268,463 rows x 39 features)

- missing cells: **0**
- infinite cells: **0**
- duplicated rows: **9,434** (3.51%)
- constant features: **0** 
- near-constant (>99.9% one value): **2** ['Telnet', 'SMTP']
- binary/category consistency: **OK**

