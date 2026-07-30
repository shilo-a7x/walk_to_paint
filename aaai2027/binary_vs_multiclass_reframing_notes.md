# Notes: Reframing the Theory from General `c` Classes to the Binary Case

## Motivation

The paper studies **signed edge prediction**, where the label space is naturally binary (`+`/`-`). While the current theory is written for an arbitrary number of classes (`c`), making the binary case the main presentation better matches the problem setting and produces cleaner information-theoretic results.

The goal is **not** to lose generality. Instead:

- Present the binary case in the main text.
- Keep the multiclass formulation as a generalization in an appendix or remark.

---

## Main Mathematical Difference

For general `c`, Fano's inequality is

```
H(Y|Z) <= H_b(P_e) + P_e log(c-1).
```

A common (but loose) lower bound is

```
P_e >= (H(Y|Z) - 1) / log(c).
```

For the binary case (`c = 2`),

```
log(c-1) = log(1) = 0,
```

so Fano simplifies to

```
H(Y|Z) <= H_b(P_e),
```

which implies

```
P_e >= H_b^{-1}(H(Y|Z)).
```

This is a much tighter and more interpretable relationship than the generic linear bound.

---

## Consequences for the Paper

Rather than presenting the theory for arbitrary `c` first:

1. Introduce binary edge labels as the primary problem setting.
2. State the binary form of Fano.
3. Develop all propositions using the binary formulation.
4. Mention that the multiclass extension follows by replacing the binary Fano inequality with its general form.
5. Move the complete multiclass derivations to the appendix.

---

## Sections That Need Reframing

### Problem Setting

Current:
- General label alphabet of size `c`.

Suggested:
- Binary labels (`+`/`-`) as the default.
- Mention that the theory extends naturally to multiclass edge prediction.

### Information-Theoretic Background

Replace the general Fano presentation with the binary version.

The multiclass statement can appear as:
- a remark,
- a theorem after the binary case,
- or in the appendix.

### Main Propositions

Replace the generic lower bounds by the binary Fano bound wherever possible.

The logical flow remains unchanged:

greater conditional entropy
→ larger unavoidable error.

Only the mathematical expression for the lower bound changes.

### Appendix

Keep:
- General notation.
- Full multiclass Fano inequality.
- General proofs.
- Statement that the binary results are a specialization.

---

## Overall Narrative

The theory should read as though it was developed specifically for signed edge prediction, with the multiclass version presented as a straightforward extension rather than the primary setting.

This better aligns the mathematical development with the experiments while preserving the generality of the theoretical framework.
