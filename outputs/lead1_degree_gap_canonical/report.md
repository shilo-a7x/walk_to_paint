# Lead 1 (GNN over-averaging) -- canonical, shared-edge degree gap

Walk model vs GNN AUC stratified by max-endpoint degree (quantile buckets),
all models on the SAME shared (walk-covered) canonical test edges, degree
from the real canonical edge list. A positive gap = walk beats the GNN in
that degree bucket. The over-averaging hypothesis predicts the gap WIDENS
with degree (more neighbors to dilute/cancel).

### bitcoin-alpha  (n_shared=2419)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 622 | [3,48] | 0.8265 | 0.7670 | +0.0595 |
| 1 | 613 | [49,110] | 0.9515 | 0.9248 | +0.0267 |
| 2 | 598 | [111,251] | 0.9330 | 0.8260 | +0.1070 |
| 3 | 586 | [254,888] | 0.9493 | 0.8888 | +0.0605 |

_low-degree gap +0.0595 -> high-degree gap +0.0605 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 622 | [3,48] | 0.8265 | 0.8138 | +0.0127 |
| 1 | 613 | [49,110] | 0.9515 | 0.9029 | +0.0485 |
| 2 | 598 | [111,251] | 0.9330 | 0.8950 | +0.0380 |
| 3 | 586 | [254,888] | 0.9493 | 0.8881 | +0.0613 |

_low-degree gap +0.0127 -> high-degree gap +0.0613 (widening = YES)_

### bitcoin-otc  (n_shared=3560)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 894 | [4,51] | 0.9407 | 0.8894 | +0.0513 |
| 1 | 892 | [52,121] | 0.9557 | 0.9151 | +0.0407 |
| 2 | 888 | [126,303] | 0.9447 | 0.8875 | +0.0572 |
| 3 | 886 | [305,1298] | 0.9310 | 0.8802 | +0.0508 |

_low-degree gap +0.0513 -> high-degree gap +0.0508 (widening = no)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 894 | [4,51] | 0.9407 | 0.8379 | +0.1029 |
| 1 | 892 | [52,121] | 0.9557 | 0.9218 | +0.0339 |
| 2 | 888 | [126,303] | 0.9447 | 0.8682 | +0.0765 |
| 3 | 886 | [305,1298] | 0.9310 | 0.8940 | +0.0370 |

_low-degree gap +0.1029 -> high-degree gap +0.0370 (widening = no)_

### epinions  (n_shared=84080)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 21239 | [1,131] | 0.8832 | 0.7924 | +0.0907 |
| 1 | 20860 | [132,325] | 0.9672 | 0.8814 | +0.0858 |
| 2 | 20971 | [327,635] | 0.9762 | 0.9002 | +0.0760 |
| 3 | 21010 | [636,3622] | 0.9766 | 0.8870 | +0.0896 |

_low-degree gap +0.0907 -> high-degree gap +0.0896 (widening = no)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 21239 | [1,131] | 0.8832 | 0.8236 | +0.0596 |
| 1 | 20860 | [132,325] | 0.9672 | 0.9332 | +0.0341 |
| 2 | 20971 | [327,635] | 0.9762 | 0.9268 | +0.0494 |
| 3 | 21010 | [636,3622] | 0.9766 | 0.9479 | +0.0286 |

_low-degree gap +0.0596 -> high-degree gap +0.0286 (widening = no)_

### wiki-elec  (n_shared=10370)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 2617 | [1,104] | 0.8969 | 0.8791 | +0.0177 |
| 1 | 2600 | [105,186] | 0.9148 | 0.8833 | +0.0315 |
| 2 | 2566 | [187,286] | 0.8901 | 0.8742 | +0.0159 |
| 3 | 2587 | [288,1167] | 0.9009 | 0.8633 | +0.0376 |

_low-degree gap +0.0177 -> high-degree gap +0.0376 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 2617 | [1,104] | 0.8969 | 0.8897 | +0.0072 |
| 1 | 2600 | [105,186] | 0.9148 | 0.9005 | +0.0143 |
| 2 | 2566 | [187,286] | 0.8901 | 0.8915 | -0.0014 |
| 3 | 2587 | [288,1167] | 0.9009 | 0.8929 | +0.0079 |

_low-degree gap +0.0072 -> high-degree gap +0.0079 (widening = YES)_

### wiki-rfa  (n_shared=17722)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 4459 | [4,127] | 0.9097 | 0.8988 | +0.0108 |
| 1 | 4424 | [128,209] | 0.8946 | 0.8723 | +0.0223 |
| 2 | 4435 | [210,345] | 0.8786 | 0.8524 | +0.0263 |
| 3 | 4404 | [347,1346] | 0.8844 | 0.8496 | +0.0348 |

_low-degree gap +0.0108 -> high-degree gap +0.0348 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 4459 | [4,127] | 0.9097 | 0.8984 | +0.0112 |
| 1 | 4424 | [128,209] | 0.8946 | 0.8814 | +0.0131 |
| 2 | 4435 | [210,345] | 0.8786 | 0.8737 | +0.0049 |
| 3 | 4404 | [347,1346] | 0.8844 | 0.8742 | +0.0101 |

_low-degree gap +0.0112 -> high-degree gap +0.0101 (widening = no)_

### slashdot090221  (n_shared=54921)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 13771 | [2,66] | 0.8716 | 0.7769 | +0.0947 |
| 1 | 13691 | [67,176] | 0.8950 | 0.7657 | +0.1293 |
| 2 | 13804 | [177,309] | 0.9028 | 0.7493 | +0.1535 |
| 3 | 13655 | [310,2557] | 0.9209 | 0.8317 | +0.0892 |

_low-degree gap +0.0947 -> high-degree gap +0.0892 (widening = no)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 13771 | [2,66] | 0.8716 | 0.8094 | +0.0622 |
| 1 | 13691 | [67,176] | 0.8950 | 0.8504 | +0.0446 |
| 2 | 13804 | [177,309] | 0.9028 | 0.8691 | +0.0337 |
| 3 | 13655 | [310,2557] | 0.9209 | 0.8826 | +0.0383 |

_low-degree gap +0.0622 -> high-degree gap +0.0383 (widening = no)_

