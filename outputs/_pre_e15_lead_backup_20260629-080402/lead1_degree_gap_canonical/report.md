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
| 0 | 622 | [3,48] | 0.8641 | 0.7670 | +0.0972 |
| 1 | 613 | [49,110] | 0.9315 | 0.9248 | +0.0068 |
| 2 | 598 | [111,251] | 0.9203 | 0.8260 | +0.0943 |
| 3 | 586 | [254,888] | 0.9192 | 0.8888 | +0.0303 |

_low-degree gap +0.0972 -> high-degree gap +0.0303 (widening = no)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 622 | [3,48] | 0.8641 | 0.8138 | +0.0503 |
| 1 | 613 | [49,110] | 0.9315 | 0.9029 | +0.0286 |
| 2 | 598 | [111,251] | 0.9203 | 0.8950 | +0.0253 |
| 3 | 586 | [254,888] | 0.9192 | 0.8881 | +0.0311 |

_low-degree gap +0.0503 -> high-degree gap +0.0311 (widening = no)_

### bitcoin-otc  (n_shared=3560)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 894 | [4,51] | 0.9319 | 0.8894 | +0.0425 |
| 1 | 892 | [52,121] | 0.9614 | 0.9151 | +0.0463 |
| 2 | 888 | [126,303] | 0.9527 | 0.8875 | +0.0651 |
| 3 | 886 | [305,1298] | 0.9366 | 0.8802 | +0.0564 |

_low-degree gap +0.0425 -> high-degree gap +0.0564 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 894 | [4,51] | 0.9319 | 0.8379 | +0.0941 |
| 1 | 892 | [52,121] | 0.9614 | 0.9218 | +0.0396 |
| 2 | 888 | [126,303] | 0.9527 | 0.8682 | +0.0845 |
| 3 | 886 | [305,1298] | 0.9366 | 0.8940 | +0.0426 |

_low-degree gap +0.0941 -> high-degree gap +0.0426 (widening = no)_

### epinions  (n_shared=73783)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 18477 | [1,125] | 0.8279 | 0.7878 | +0.0401 |
| 1 | 18432 | [126,317] | 0.9447 | 0.8819 | +0.0628 |
| 2 | 18478 | [318,628] | 0.9563 | 0.9016 | +0.0547 |
| 3 | 18396 | [629,3622] | 0.9654 | 0.8947 | +0.0707 |

_low-degree gap +0.0401 -> high-degree gap +0.0707 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 18477 | [1,125] | 0.8279 | 0.8123 | +0.0156 |
| 1 | 18432 | [126,317] | 0.9447 | 0.9322 | +0.0125 |
| 2 | 18478 | [318,628] | 0.9563 | 0.9247 | +0.0316 |
| 3 | 18396 | [629,3622] | 0.9654 | 0.9445 | +0.0209 |

_low-degree gap +0.0156 -> high-degree gap +0.0209 (widening = YES)_

### wiki-elec  (n_shared=8857)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 2254 | [1,100] | 0.8889 | 0.8738 | +0.0151 |
| 1 | 2213 | [101,175] | 0.9090 | 0.8908 | +0.0181 |
| 2 | 2224 | [176,280] | 0.8846 | 0.8708 | +0.0138 |
| 3 | 2166 | [281,1167] | 0.8879 | 0.8580 | +0.0299 |

_low-degree gap +0.0151 -> high-degree gap +0.0299 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 2254 | [1,100] | 0.8889 | 0.8824 | +0.0065 |
| 1 | 2213 | [101,175] | 0.9090 | 0.9041 | +0.0049 |
| 2 | 2224 | [176,280] | 0.8846 | 0.8871 | -0.0025 |
| 3 | 2166 | [281,1167] | 0.8879 | 0.8866 | +0.0013 |

_low-degree gap +0.0065 -> high-degree gap +0.0013 (widening = no)_

### wiki-rfa  (n_shared=15280)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 3831 | [4,128] | 0.8903 | 0.8982 | -0.0079 |
| 1 | 3864 | [129,213] | 0.8858 | 0.8818 | +0.0040 |
| 2 | 3777 | [214,351] | 0.8671 | 0.8569 | +0.0102 |
| 3 | 3808 | [353,1346] | 0.8767 | 0.8497 | +0.0271 |

_low-degree gap -0.0079 -> high-degree gap +0.0271 (widening = YES)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 3831 | [4,128] | 0.8903 | 0.8949 | -0.0045 |
| 1 | 3864 | [129,213] | 0.8858 | 0.8812 | +0.0046 |
| 2 | 3777 | [214,351] | 0.8671 | 0.8745 | -0.0075 |
| 3 | 3808 | [353,1346] | 0.8767 | 0.8685 | +0.0082 |

_low-degree gap -0.0045 -> high-degree gap +0.0082 (widening = YES)_

### slashdot090221  (n_shared=53962)

**walk_full vs GINEConv**

| bucket | n | deg range | walk AUC | GINEConv AUC | gap (walk-GINEConv) |
|---|---|---|---|---|---|
| 0 | 13569 | [2,65] | 0.8661 | 0.7776 | +0.0885 |
| 1 | 13471 | [66,175] | 0.8892 | 0.7661 | +0.1231 |
| 2 | 13467 | [176,309] | 0.8951 | 0.7502 | +0.1449 |
| 3 | 13455 | [310,2557] | 0.9159 | 0.8325 | +0.0834 |

_low-degree gap +0.0885 -> high-degree gap +0.0834 (widening = no)_

**walk_full vs SiGAT**

| bucket | n | deg range | walk AUC | SiGAT AUC | gap (walk-SiGAT) |
|---|---|---|---|---|---|
| 0 | 13569 | [2,65] | 0.8661 | 0.8084 | +0.0577 |
| 1 | 13471 | [66,175] | 0.8892 | 0.8482 | +0.0409 |
| 2 | 13467 | [176,309] | 0.8951 | 0.8681 | +0.0270 |
| 3 | 13455 | [310,2557] | 0.9159 | 0.8806 | +0.0353 |

_low-degree gap +0.0577 -> high-degree gap +0.0353 (widening = no)_

