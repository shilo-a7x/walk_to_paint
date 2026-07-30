# Framing Shift: Transitioning $c > 2$ to $c = 2$ in Pewter Theory

## 1. The Core Mathematical Issue
In the general $c$-class formulation, Fano's inequality reads:
$$H(Y \mid Z) \le H_b(P_e) + P_e \log_2(c - 1)$$

When specializing to binary edge sign prediction ($c = 2$):
* $\log_2(c - 1) = \log_2(1) = 0$.
* The linear term $P_e \log_2(c - 1)$ **vanishes entirely**.
* Fano's inequality simplifies to exact binary entropy matching:
  $$H(Y \mid Z) \le H_b(P_e)$$
  where $H_b(P_e) = -P_e \log_2 P_e - (1 - P_e) \log_2(1 - P_e)$.

### Why the standard formula fails for $c = 2$
If you naively plug $c = 2$ into the standard relaxed bound $P_e \ge \frac{H(Y \mid Z) - 1}{\log_2 c}$, you get:
$$P_e \ge H(Y \mid Z) - 1$$
Since $H(Y \mid Z) \le 1$ for binary variables, this yields $P_e \ge \text{negative number}$ or $P_e \ge 0$, which is **vacuous** and loses the conditional entropy bound completely.

---

## 2. Solutions: How to Keep the Information Bound Tight

### Option A: Inverse Binary Entropy Form (Exact & Elegant)
For $P_e \le 0.5$ (better than random guessing on binary classification):
$$P_e \ge H_b^{-1}\big(H(Y \mid Z)\big)$$
* **Pros:** Exact, mathematically pristine.
* **Cons:** Requires defining $H_b^{-1}$ on $[0, 0.5]$.

### Option B: Closed-Form Quadratic Bound (Explicit & Direct)
Using the standard approximation $H_b(P_e) \le 1 - \frac{2}{\ln 2}\left(\frac{1}{2} - P_e\right)^2$, we get an explicit lower bound:
$$P_e \ge \frac{1}{2} - \sqrt{\frac{\ln 2}{2} \Big(1 - H(Y \mid Z)\Big)}$$
* **Pros:** Shows explicitly how error deviates from random choice ($0.5$) as conditional entropy $H(Y \mid Z)$ decreases from $1$ bit.

### Option C: Unified Piecewise Statement (Best for Paper Structure)
State the exact binary case first (since $c=2$ is the paper's primary setting), followed by the general $c > 2$ linear relaxation as a natural generalization:

$$
P_e \ge 
\begin{cases}
H_b^{-1}\Big(H\big(Y \mid c(U), c(V)\big)\Big), & \text{if } c = 2 \text{ (exact binary sign prediction)} \\[8pt]
\dfrac{H\big(Y \mid c(U), c(V)\big) - 1}{\log_2 c}, & \text{for general } c > 2 \text{ (multi-class edge labels)}
\end{cases}
$$

---

## 3. Recommended Reframing / Rewrite Notes

1. **Lead with $c = 2$ in Prose:**
   * Motivate edge sign prediction directly ($+$ vs. $-$ trust, activation vs. inhibition).
   * Note immediately that binary labels simplify Fano's bound directly to $H_b(P_e)$, making the entropy-error relationship tighter and cleaner than in $c > 2$.

2. **Update Definitions & Notation:**
   * Express entropy explicitly in **bits** ($\log_2$).
   * Explicitly state that $c=2$ gives the sharpest bound, while the $c>2$ formula provides the multi-class extension without losing theoretical rigor.

3. **In Propositions 1 & 2:**
   * Use the piecewise formulation or present $c=2$ as the primary theorem, adding a brief remark or corollary for general $c$.
   * Highlight that in binary prediction, $H(Y \mid Z) \to 1$ forces $P_e \to 0.5$ (pure coin flip), directly establishing why high endpoint entropy degrades GNN performance to random chance.
