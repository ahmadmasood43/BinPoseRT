# Alpha results — tless

AR in %, BOP19 protocol. *core* = `binposert.evaluate` (development metric); *official* = `bop_toolkit` via `tools/bop_eval.sh` (reported number).

| exp | seg | coarse pose | AR core | AR official | VSD | MSSD | MSPD | n pred | run |
|---|---|---|---|---|---|---|---|---|---|
| A0 | gt | foundpose | 59.0 | 59.2 | 45.7 | 50.1 | 81.9 | 6423 | `outputs/tless/test_primesense/evaluate/177a35975045c1e6` |
| A1 | cnos | foundpose | 35.2 | 35.4 | 27.4 | 30.1 | 48.6 | 5452 | `outputs/tless/test_primesense/evaluate/8c0fce8cb8bdd94d` |
| A5 | cnos | megapose | 48.0 | 48.2 | 45.6 | 45.9 | 53.0 | 5458 | `outputs/tless/test_primesense/evaluate/1c7d4f12d2aaf288` |

## AR by visible fraction (core metric, pooled over GT)

| exp | [0.1, 0.3) n | [0.3, 0.6) n | [0.6, 1.0) n |
|---|---|---|---|
| A0 | 15.1 (159) | 33.7 (702) | 63.4 (5562) |
| A1 | 1.0 (159) | 7.4 (702) | 39.7 (5562) |
| A5 | 0.9 (159) | 10.0 (702) | 54.2 (5562) |

Commits: A0=0c9b3498ae45d5af0122955f8780c85cee128283, A1=0c9b3498ae45d5af0122955f8780c85cee128283, A5=0c9b3498ae45d5af0122955f8780c85cee128283

