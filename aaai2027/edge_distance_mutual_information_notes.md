# Edge Distance and Mutual Information Notes

## Key ambiguity

There is **no canonical distance between edges** in a directed graph.
The right definition depends on what "information propagation" means.

## Important observation

Consider

    u → v ← x → y

Even though there is no directed edge-to-edge walk from `(u,v)` to
`(x,y)`, there **is** a dependency:

-   node `v` aggregates information from both `u` and `x`;
-   edge `(x,y)` carries information originating at `x`;
-   therefore `(u,v)` and `(x,y)` may be statistically related through
    the latent state of node `x`.

Thus, using only directed edge walks may underestimate available
information.

## Three notions of locality

### 1. Directed walk locality

Edge distance follows consecutive directed edges. Pros: matches path
traversal. Cons: sibling/zig-zag edges become unreachable.

### 2. Vertex-shell locality

For query edge `(u,v)`, perform BFS from `{u,v}` and assign edges
according to the minimum distance of either endpoint. Pros: mirrors
k-hop GNN neighborhoods.

### 3. Information locality (recommended)

Instead of defining edge distance axiomatically, define the information
source.

For a query edge `(u,v)`, define the k-hop vertex neighborhood around
its endpoints. All edges incident to those vertices belong to shell k.

This reflects that information is stored in node representations, not in
edge adjacency alone.

## Mutual information

Measure

`I(Y_e ; Y_f)` averaged over edges `f` in shell `k`

or

`I(Y_e ; {Y_f : f in shell_k})`

and study decay versus `k`.

## Takeaway

If the paper argues about GNN information flow, vertex-shell distance is
preferable because message passing is vertex-centric. If the paper
argues about directed path propagation, line-graph distance is
appropriate. They answer different scientific questions.
