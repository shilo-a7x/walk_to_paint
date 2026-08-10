# Canonical-split per-edge join (proof of shared ground truth)

All models in raw (u,v) space. `overlap` = walk_full ∩ GINEConv ∩ SiGAT edges.

| dataset | walk_full | GINEConv | SiGAT | overlap | walk AUC@ovl | GINE AUC@ovl | SiGAT AUC@ovl |
|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 2419 | 2419 | 2419 | 2419 | 0.9237 | 0.8610 | 0.8821 |
| bitcoin-otc | 3560 | 3560 | 3560 | 3560 | 0.9421 | 0.8849 | 0.8759 |
| wiki-elec | 10370 | 10370 | 10370 | 10370 | 0.9008 | 0.8753 | 0.8930 |
| wiki-rfa | 17722 | 17722 | 17722 | 17722 | 0.8931 | 0.8699 | 0.8831 |
| slashdot090221 | 54921 | 54921 | 54921 | 54921 | 0.9007 | 0.7864 | 0.8588 |
| epinions | 84080 | 84080 | 84080 | 84080 | 0.9557 | 0.8610 | 0.9137 |
