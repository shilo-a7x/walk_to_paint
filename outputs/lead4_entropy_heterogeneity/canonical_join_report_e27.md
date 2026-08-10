# Canonical-split per-edge join, UPDATED walk model (edge_cover + LocalAttn4, E25/E26/E27)

All models in raw (u,v) space. `overlap` = walk_full ∩ GINEConv ∩ SiGAT edges.

| dataset | walk_full | GINEConv | SiGAT | overlap | walk AUC@ovl | GINE AUC@ovl | SiGAT AUC@ovl |
|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 2419 | 2419 | 2419 | 2419 | 0.9226 | 0.8610 | 0.8821 |
| bitcoin-otc | 3560 | 3560 | 3560 | 3560 | 0.9312 | 0.8849 | 0.8759 |
| wiki-elec | 10370 | 10370 | 10370 | 10370 | 0.9032 | 0.8753 | 0.8930 |
| wiki-rfa | 17722 | 17722 | 17722 | 17722 | 0.8928 | 0.8699 | 0.8831 |
| slashdot090221 | 54921 | 54921 | 54921 | 54921 | 0.9004 | 0.7864 | 0.8588 |
| epinions | 84080 | 84080 | 84080 | 84080 | 0.9507 | 0.8610 | 0.9137 |
