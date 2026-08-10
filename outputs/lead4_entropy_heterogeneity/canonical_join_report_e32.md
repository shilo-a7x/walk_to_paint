# Canonical-split per-edge join, POST-MIGRATION walk model (E31/E32)

All models in raw (u,v) space. `overlap` = walk_localattn4 ∩ GINEConv ∩ SiGAT edges.

| dataset | walk_localattn4 | GINEConv | SiGAT | overlap | walk AUC@ovl | GINE AUC@ovl | SiGAT AUC@ovl |
|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 2419 | 2419 | 2419 | 2419 | 0.9185 | 0.8610 | 0.8821 |
| bitcoin-otc | 3560 | 3560 | 3560 | 3560 | 0.9360 | 0.8849 | 0.8760 |
| wiki-elec | 10370 | 10370 | 10370 | 10370 | 0.9075 | 0.8753 | 0.8930 |
| wiki-rfa | 17722 | 17722 | 17722 | 17722 | 0.8975 | 0.8699 | 0.8831 |
| slashdot090221 | 54921 | 54921 | 54921 | 54921 | 0.8980 | 0.7864 | 0.8587 |
| epinions | 84080 | 84080 | 84080 | 84080 | 0.9527 | 0.8610 | 0.9146 |
