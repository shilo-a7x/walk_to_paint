# Lead 4c — all equations

Every quantity and model used by `scripts/lead4c_entropy_logit_regression.py`, in one place.
Ground truth throughout: the shared (E15 full-coverage) edge set, same 4 models
(`walk_full, walk_localattn4, GINEConv, SiGAT`) on identical edges per dataset.

---

## 0. Notation

- Directed signed graph $G=(V,E)$; edge $(u\to v)$ has sign $s\in\{+1,-1\}$ (stored as $y\in\{0,1\}$).
- We predict the sign of edge $i=(u_i\to v_i)$. Model outputs $\hat p_i=P(\text{positive})$.
- **Correctness** (the regression outcome):
$$\text{correct}_i = \mathbb{1}\big[\,\mathbb{1}[\hat p_i\ge 0.5]=y_i\,\big]\in\{0,1\}.$$
- **Binary Shannon entropy** of a Bernoulli fraction $p$ (in bits):
$$H(p) = -p\log_2 p-(1-p)\log_2(1-p),\qquad H(0)=H(1)=0.$$
$H=0$ ⟹ perfectly homogeneous signs; $H=1$ ⟹ 50/50 (maximally contested).

---

## 1. Per-node directional sign-entropy (the atoms)

For a node $n$ define three incident edge-sign multisets:
- $\mathrm{out}(n)$ = signs of edges where $n$ is the **source**,
- $\mathrm{in}(n)$ = signs of edges where $n$ is the **target**,
- $\mathrm{inout}(n)=\mathrm{out}(n)\cup\mathrm{in}(n)$.

With positive-fraction $p_{\mathrm{dir}}(n)=\dfrac{|\{s>0\}\cap \mathrm{dir}(n)|}{|\mathrm{dir}(n)|}$, the node entropy is
$$H_{\mathrm{dir}}(n)=H\big(p_{\mathrm{dir}}(n)\big),\qquad \mathrm{dir}\in\{\mathrm{out},\mathrm{in},\mathrm{inout}\}.$$

For an edge $(u\to v)$ the **four atomic node terms** are
$$\texttt{src\_out}=H_{\mathrm{out}}(u),\quad \texttt{src\_in}=H_{\mathrm{in}}(u),\quad \texttt{tgt\_out}=H_{\mathrm{out}}(v),\quad \texttt{tgt\_in}=H_{\mathrm{in}}(v).$$

---

## 2. Two-hop path-consistency entropy

Given an adjacency $A$ (either out-adjacency or in-adjacency), for an anchor node $n$ enumerate all 2-hop paths $n\to m\to k$. A path is **consistent** if the two hop-signs agree, $\mathrm{sign}(n\to m)=\mathrm{sign}(m\to k)$. Let
$$T(n)=\#\{\text{2-hop paths from } n\},\qquad C(n)=\#\{\text{consistent 2-hop paths from } n\},\qquad q(n)=\frac{C(n)}{T(n)}.$$
The two-hop entropy is $H(q(n))$. Three variants for edge $(u\to v)$:

$$
\texttt{twohop\_out}=H\!\big(q^{\mathrm{out}}(v)\big)\ \ (v\!\to\! m\!\to\! k),\qquad
\texttt{twohop\_in}=H\!\big(q^{\mathrm{in}}(u)\big)\ \ (s\!\to\! t\!\to\! u),
$$
$$
\texttt{twohop\_inout}=H\!\left(\frac{C^{\mathrm{out}}(v)+C^{\mathrm{in}}(u)}{T^{\mathrm{out}}(v)+T^{\mathrm{in}}(u)}\right)\quad(\textbf{counts pooled before the entropy}).
$$

---

## 3. Pre-atomic model = the "marginal3" 12-combo sweep (`spec="marginal3"`)

A **node-variant** chooses a direction for the source side and the target side; a **two-hop variant** chooses one $H_{2}$. The four node-variants are
$$\texttt{out\_out}=(\mathrm{out},\mathrm{out}),\ \ \texttt{in\_in}=(\mathrm{in},\mathrm{in}),\ \ \texttt{out\_in}=(\mathrm{out},\mathrm{in}),\ \ \texttt{inout\_inout}=(\mathrm{inout},\mathrm{inout}),$$
and the two-hop variants are $\{\mathrm{out},\mathrm{in},\mathrm{inout}\}$ → $4\times3=12$ combos. For node-variant $(\delta_s,\delta_t)$ and two-hop variant $\tau$:

$$\operatorname{logit}P(\text{correct}_i)=\beta_0+\beta_{\text{src}}\,H_{\delta_s}(u_i)+\beta_{\text{tgt}}\,H_{\delta_t}(v_i)+\beta_{\text{2hop}}\,H_{2,\tau}(i).$$

Each of the 12 combos is fit independently (3 betas each). **Note:** `inout`-variants pool out+in at the *count* level (§1–2), not by averaging entropies. The combos are overlapping pairings of the same six atoms (e.g. `in_in`, `out_in`, `inout_inout` all reuse $H_{\mathrm{in}}(v)=\texttt{tgt\_in}$).

---

## 4. Atomic decomposition (Phase-1 headline, `spec="atomic"`)

All six atoms entered **simultaneously**, one partial coefficient each:

$$\operatorname{logit}P(\text{correct}_i)=\beta_0+\beta_{so}\,\texttt{src\_out}_i+\beta_{si}\,\texttt{src\_in}_i+\beta_{to}\,\texttt{tgt\_out}_i+\beta_{ti}\,\texttt{tgt\_in}_i+\beta_{2i}\,\texttt{twohop\_in}_i+\beta_{2o}\,\texttt{twohop\_out}_i.$$

"Which direction matters" is read off the $\beta$'s; no direction is privileged and no combo is a designated control. An edge is dropped if **any** of the six is undefined (common-support / listwise deletion).

---

## 5. Compact composite (`spec="composite"`) — one node-β + one path-β

Two predictors, built by **count-level pooling** (NOT by averaging variant entropies):

$$\texttt{b\_node}_i = H\!\left(\frac{\mathrm{pos}_{\mathrm{inout}}(u_i)+\mathrm{pos}_{\mathrm{inout}}(v_i)}{\mathrm{tot}_{\mathrm{inout}}(u_i)+\mathrm{tot}_{\mathrm{inout}}(v_i)}\right),\qquad \texttt{b\_path}_i = \texttt{twohop\_inout}_i\ (\text{§2}),$$

where $\mathrm{pos}_{\mathrm{inout}}(n),\mathrm{tot}_{\mathrm{inout}}(n)$ are the positive and total counts of all edges incident to $n$. The model:

$$\operatorname{logit}P(\text{correct}_i)=\beta_0+b_{\text{node}}\,\texttt{b\_node}_i+b_{\text{path}}\,\texttt{b\_path}_i.$$

**Caveat:** `b_node` pools `src_out` (where the walk is hurt more) with `tgt_in` (where GNNs are hurt more), so it *cancels* the src/tgt asymmetry — a companion to §4, not a replacement.

---

## 6. Pooling specs (per-dataset / shared-slope / interacted)

Let $X_i$ be the slope-feature vector of whichever model (§3/§4/§5), $D$ datasets.

- **Per-dataset:** fit the model separately on each dataset $d$ (its own $\beta_0,\beta$).
- **Pooled, shared slope** (one $\beta$ for all datasets, datasets shift only the intercept via fixed effects):
$$\operatorname{logit}P(\text{correct}_i)=\beta_0+\sum_{d\ne d_0}\gamma_d\,\mathbb{1}[\text{ds}_i=d]+\beta^\top X_i.$$
- **Pooled, interacted** (marginal3 only — each dataset gets its **own** slope; equivalent to the per-dataset fits):
$$\operatorname{logit}P(\text{correct}_i)=\beta_0+\sum_{d\ne d_0}\gamma_d\,\mathbb{1}[\text{ds}_i=d]+\beta^\top X_i+\sum_{d\ne d_0}\big(\delta_d^\top X_i\big)\mathbb{1}[\text{ds}_i=d].$$
"Shared vs interacted" = "is the slope the same in every dataset, or dataset-specific?" (the caterpillar figure overlays the two).

**Standardized betas** $\beta^{\text{std}}$ come from refitting on $z$-scored features $\tilde X=(X-\bar X)/\mathrm{sd}(X)$ (for cross-term comparability); raw $\beta$ are reported with CIs.

---

## 7. Inference

- **Two-way cluster-robust covariance** (Cameron–Gelbach–Miller 2011), clustering on source $u$ and target $v$ (edges sharing an endpoint aren't independent):
$$\hat V_{\text{2way}}=\hat V_u+\hat V_v-\hat V_{u\cap v},\qquad \mathrm{SE}_{\text{robust}}=\sqrt{\operatorname{diag}\hat V_{\text{2way}}}.$$
(Falls back to one-way on $u$ if $\hat V_{\text{2way}}$ is not PSD-safe.) Then
$$z=\frac{\beta}{\mathrm{SE}_{\text{robust}}},\qquad p=2\big(1-\Phi(|z|)\big),\qquad \text{95\% CI}=\beta\pm1.96\,\mathrm{SE}_{\text{robust}}.$$
- **Odds ratio:** $\mathrm{OR}=e^{\beta}$ (multiplicative effect on the odds of a correct prediction per **+1 bit** of entropy).
- **McFadden pseudo-$R^2$:** $1-\ell_{\text{full}}/\ell_{\text{null}}$.
- **Benjamini–Hochberg FDR** within each family (per-dataset and pooled-shared-slope, separately per `spec`): for sorted $p_{(1)}\le\dots\le p_{(n)}$,
$$p^{\text{FDR}}_{(k)}=\min_{j\ge k}\ \min\!\Big(1,\ \tfrac{n}{j}\,p_{(j)}\Big).$$ A result is "significant" iff $p^{\text{FDR}}<0.05$ (filled marker / `*`).

---

## 8. Model-free cross-checks (no logistic regression)

- **Spearman rank correlation** $\rho$ between each entropy term $X$ and $\text{correct}$, per model (sign should match the regression $\beta$).
- **Binned mutual information** (bits): bin $X$ into quantile bins $\tilde X$; with binary $Y=\text{correct}$,
$$\mathrm{MI}(\tilde X;Y)=\sum_{x}\sum_{y\in\{0,1\}}p(x,y)\,\log_2\frac{p(x,y)}{p(x)\,p(y)}.$$

---

## 9. Cross-dataset consistency scorecard

For a term, compare the mean GNN $\beta$ vs mean walk $\beta$ per dataset; the score is the fraction of the 6 datasets with $\bar\beta_{\text{GNN}}<\bar\beta_{\text{walk}}$, tested against chance with a two-sided **sign test** (binomial vs $0.5$). With $n=6$ only $0\%$ or $100\%$ reaches $p\approx0.031$.
