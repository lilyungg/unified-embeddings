# Feature Multiplexing under Retrieval Losses: Gradient Analysis

This note ports the gradient analysis of Coleman et al. (2023), §4.2 — done
there for binary logistic regression (pointwise log-loss) — to the two-tower
candidate-generation setting with softmax-family losses. We derive the analog
of their gradient decomposition (their Eqs. (3)–(5)), the matrix generalization
of the weight-orthogonalization condition, and show where and why the
$O(N/M)$ norm-growth prediction breaks under softmax. Empirical status of each
claim is in README ("Candidate generation").

## 1. Setup and notation

**Features and vocabularies.** User features $t \in [T]$ with vocabularies
$V_t$, $N_t = |V_t|$; an item vocabulary $I$ with $n = |I|$. Write
$\tilde V = V_1 \sqcup \dots \sqcup V_T \sqcup I$ for the disjoint (feature-salted)
union, $N = |\tilde V| = \sum_t N_t + n$.

**Shared (multiplexed) table.** $E \in \mathbb{R}^{M \times d}$ with a single
2-universal hash $h : \tilde V \to [M]$ (feature salting makes the restrictions
$h|_{V_t}$, $h|_I$ independent). For $a \in \tilde V$ write $e_a := E_{h(a)}$.
Collision indicator: $\mathbb{1}_{a,b} := \mathbb{1}\{h(a) = h(b)\}$,
$\mathbb{E}[\mathbb{1}_{a,b}] = 1/M$ for $a \neq b$.

**Model (linear two-tower).** Per-feature *readers* $A_t \in \mathbb{R}^{k \times d}$
and an item reader $B \in \mathbb{R}^{k \times d}$:

$$u(x) = \sum_{t=1}^{T} A_t\, e_{x_t} \in \mathbb{R}^k, \qquad
  v_j = B\, e_j \in \mathbb{R}^k \ (j \in I), \qquad
  s_j(x) = \langle u(x), v_j \rangle .$$

This is the direct generalization of the paper's single-layer model
$z = \sum_t \langle \theta_t, e_{x_t}\rangle$: their reader is a vector
$\theta_t \in \mathbb{R}^d$ (the case $k=1$); ours is a matrix because the
tower output is a $k$-vector, not a scalar.

**Data.** $D = \{(x^{(i)}, y^{(i)})\}$, $x = (x_1,\dots,x_T)$, positive item
$y \in I$.

**Loss (full softmax).**

$$L(x,y) = -s_y(x) + \log \sum_{j \in I} e^{s_j(x)}, \qquad
  p_j = \frac{e^{s_j}}{\sum_{l} e^{s_l}} .$$

**Residuals.** $r_j := \partial L / \partial s_j = p_j - \delta_{j,y}$.
This is the analog of the BCE residual $\rho = \sigma(z) - y$; the structural
difference is:

> **Lemma 1 (zero-sum residuals).** $\sum_{j \in I} r_j = \sum_j p_j - 1 = 0.$

BCE has no such constraint: its residual is one free scalar per example.

**Context vector.** $q(x,y) := \sum_{j} r_j v_j
= \mathbb{E}_{j \sim p}[v_j] - v_y$ — the probability-weighted mean item vector
minus the positive item vector. Then $\partial L/\partial u = q$ and
$\partial L/\partial v_j = r_j\, u$.

## 2. Gradient decomposition of a shared row

By the chain rule, for a table row index $m \in [M]$ and a single example:

$$\nabla_{E_m} L(x,y)
 = \sum_{t=1}^{T} \mathbb{1}\{h(x_t) = m\}\, A_t^{\!\top} q(x,y)
 \;+\; \sum_{j \in I} \mathbb{1}\{h(j) = m\}\, r_j(x,y)\, B^{\!\top} u(x). \tag{1}$$

Fix a reference value $w \in V_1$ and let $m = h(w)$. Summing (1) over $D$ and
grouping by who hashed into the row (exactly as the paper does when passing
from their $\nabla_{E_m}$ to $\nabla_{e_{h(u)}}$):

$$\nabla_{E_m} L_D \;=\;
\underbrace{\sum_{(x,y):\, x_1 = w} A_1^{\!\top} q(x,y)}_{\text{(T) true}}
\;+\;
\underbrace{\sum_{\substack{w' \in V_1 \setminus \{w\}}} \mathbb{1}_{w,w'}
    \sum_{(x,y):\, x_1 = w'} A_1^{\!\top} q(x,y)}_{\text{(A) intra-feature}} $$

$$\;+\;
\underbrace{\sum_{t' \neq 1} \sum_{w'' \in V_{t'}} \mathbb{1}_{w,w''}
    \sum_{(x,y):\, x_{t'} = w''} A_{t'}^{\!\top} q(x,y)}_{\text{(E) inter-feature (user side)}}
\;+\;
\underbrace{\sum_{j \in I} \mathbb{1}_{w,j}
    \sum_{(x,y)} r_j(x,y)\, B^{\!\top} u(x)}_{\text{(X) cross-tower}} . \tag{2}$$

Correspondence with the paper: (T), (A), (E) are the exact analogs of their
Eqs. (3), (4), (5), with the scalar weights
$C_{u,v,0} - (C_{u,v,0}+C_{u,v,1})\sigma_\theta(e_{u,v})$ replaced by the
softmax context $q(x,y)$, and the directions $\theta_1, \theta_2$ replaced by
the row spaces $\mathrm{range}(A_1^{\!\top}), \mathrm{range}(A_{t'}^{\!\top})$.
Term (X) is new: it appears because the item tower shares the table.

**Directional structure.**
- (T) and (A) both lie in $\mathrm{range}(A_1^{\!\top})$. At readout by feature 1
  they are indistinguishable — intra-feature collisions act as value merging and
  are unrecoverable, exactly as in the paper.
- (E) lies in $\bigoplus_{t' \neq 1} \mathrm{range}(A_{t'}^{\!\top})$,
  (X) lies in $\mathrm{range}(B^{\!\top})$ — *other readers' subspaces*.

The choice of loss only changes the scalar/vector weights inside the sums
($q$ and $r_j$); the four-way directional structure follows from linearity of
the lookup and the chain rule alone. This is why the decomposition transfers
to any loss that touches embeddings through linear readers.

## 3. Readout annihilation and the orthogonality condition

Following the paper (their footnote 2), freeze the readers during training and
consider the gradient-accumulated part of a row. By (2), after any number of
SGD steps,

$$E_m \in \mathrm{range}(A_1^{\!\top}) + \sum_{t' \neq 1}\mathrm{range}(A_{t'}^{\!\top})
       + \mathrm{range}(B^{\!\top}),$$

i.e. $E_m = A_1^{\!\top} a + \sum_{t'\neq 1} A_{t'}^{\!\top} b_{t'} + B^{\!\top} c$
for some coefficient vectors $a, b_{t'}, c \in \mathbb{R}^k$. Feature 1 reads
the row as

$$A_1 E_m \;=\; (A_1 A_1^{\!\top})\, a
 \;+\; \sum_{t' \neq 1} (A_1 A_{t'}^{\!\top})\, b_{t'}
 \;+\; (A_1 B^{\!\top})\, c. \tag{3}$$

> **Proposition 2 (annihilation).** The inter-feature and cross-tower
> contamination in (3) vanishes for **all** data and training histories iff
>
> $$A_1 A_{t'}^{\!\top} = 0 \ \ (\forall t' \neq 1) \qquad\text{and}\qquad
>   A_1 B^{\!\top} = 0. \tag{$\star$}$$
>
> *Proof.* Sufficiency is immediate from (3). Necessity: if
> $A_1 A_{t'}^{\!\top} \neq 0$, pick $b_{t'}$ with
> $(A_1 A_{t'}^{\!\top}) b_{t'} \neq 0$; a collision pattern and dataset
> realizing a multiple of $A_{t'}^{\!\top} b_{t'}$ in the row exists with
> nonzero probability. $\square$

Condition $(\star)$ generalizes the paper's $\langle\theta_1,\theta_2\rangle = 0$:
$A_1 A_{t'}^{\!\top} = 0$ says every row of $A_1$ is orthogonal to every row of
$A_{t'}$, i.e. the *hearing subspaces* (row spaces) are mutually orthogonal.
With $k = 1$ it reduces literally to $\theta_1^{\!\top}\theta_2 = 0$.

**Measurement.** A single angle between two subspaces is not defined; we track
the total leakage energy

$$\mathrm{ov}(A,B) := \frac{\lVert A B^{\!\top} \rVert_F}{\lVert A \rVert_F \lVert B \rVert_F}
 \in [0, 1],
 \qquad \lVert A B^{\!\top}\rVert_F^2 = \sum_{i,j} \langle a_i, b_j \rangle^2,$$

a singular-value-weighted average of $\cos^2$ of the principal angles.
Calibration matters: for full-rank Gaussian $32{\times}30$ blocks,
$\mathrm{ov} \approx 0.26$ for *identical* blocks and
$\mathrm{ov} \approx 1/\sqrt{d} \approx 0.183$ for *independent random* blocks;
$\mathrm{ov} = 1$ is attained only by rank-one blocks. (Averaging the rows of
each block and comparing two mean vectors is **not** a valid proxy: rows may
cancel — blocks $\{x, y\}$ and $\{x, -y\}$ have orthogonal means yet
$\mathrm{ov} = 0.707$.)

> **Corollary 3 (rank budget).** If $(\star)$ holds for all pairs
> (including $B$), the row spaces are mutually orthogonal subspaces of
> $\mathbb{R}^d$, hence
>
> $$\sum_{t=1}^{T} \mathrm{rank}(A_t) + \mathrm{rank}(B) \;\le\; d. \tag{4}$$

With $T{+}1 = 6$ readers and $d = 30$: at most $5$ directions per reader on
average. In the paper's scalar case (4) is the trivial $T \le d$; here it is a
substantive prediction: under heavy collision pressure readers must trade
effective rank for orthogonality (*rank collapse*; untested yet).

**Why training moves toward $(\star)$** (heuristic, at the paper's level of
rigor): while $(\star)$ fails, the contamination terms in (3) inject
data-independent noise into every score $s_j$ through $u(x)$; the expected
excess loss is quadratic in the leakage operators $A_1 A_{t'}^{\!\top}$,
$A_1 B^{\!\top}$, with a prefactor proportional to the collision mass
($\propto 1/M$ times co-occurrence counts). Gradient descent on the readers
therefore has a descent direction that shrinks leakage wherever collisions
exist, and no such pressure when they do not (collisionless control).

## 4. Norm dynamics: where softmax breaks $O(N/M)$

**Recap of the paper's argument (BCE).** After orthogonalization, the
inter-feature components (their Eq. (5)) are annihilated *at readout*: the loss
can no longer see them, so nothing corrects them. Each colliding value
delivers an $O(1)$ increment upon each appearance in the data; a row hosting
$\approx N/M$ values receives that many independent, uncorrected increment
streams. Uncorrected streams accumulate diffusively, so
$\mathbb{E}\lVert E_m \rVert^2 = \Theta(N/M)$ — their Fig. 2, our ranking
replication ($\times 7.4$ at $0.1\times$).

**Softmax, item side.** Let $S = \{ j \in I : h(j) = m \}$ be the items
sharing row $m$, occupancy $|S| \approx n/M$. From (1), a single example
contributes to the row (item-side part):

$$\nabla_{E_m}^{\text{item}} L(x,y)
 = B^{\!\top} u(x) \sum_{j \in S} r_j
 = B^{\!\top} u(x)\, \big( P_S(x) - \mathbb{1}\{y \in S\} \big),
 \qquad P_S(x) := \sum_{j \in S} p_j(x). \tag{5}$$

> **Lemma 4 (bounded bucket residual).** For any occupancy $|S|$,
> $\big| P_S(x) - \mathbb{1}\{y \in S\} \big| \le 1$.
>
> *Proof.* $0 \le P_S \le \sum_{j\in I} p_j = 1$ by Lemma 1. $\square$

This is the first mechanism: however many items merge into a row, one example
hands the row a **single scalar in $[-1,1]$** times $B^{\!\top}u$. Under BCE
the number of $O(1)$ pushes a row receives scales with its occupancy; under
softmax it does not — the zero-sum constraint aggregates the bucket's residual
mass before it reaches the parameters.

> **Proposition 5 (bucket-level stationarity).** The item-side gradient of row
> $m$ vanishes in expectation iff
>
> $$\mathbb{E}_{(x,y)}\big[ \big( P_S(x) - \mathbb{1}\{y \in S\} \big)\, u(x) \big] = 0, \tag{6}$$
>
> which holds in particular under pointwise bucket calibration
> $P_S(x) = \Pr[\,y \in S \mid x\,]$. The bucket behaves as a single merged
> item of frequency $f_S = \sum_{j \in S} f_j$; identities within the bucket
> receive no gradient signal (value merging, as in the paper), and once the
> aggregate is calibrated the restoring force on the row dies.

Two consequences for norms:

1. **The visible pathway is error-corrected.** Unlike the paper's invisible
   post-orthogonalization components, (5) is the gradient of the loss through a
   *visible* readout ($B$ reads the row directly), so it has a restoring force
   toward the stationary point (6) — it converges rather than diffuses.
2. **The stationary norm grows at most logarithmically in occupancy.** Scores
   enter softmax through $\exp$: for a merged item to carry probability mass
   $f_S \propto |S|$, its score needs $s^* \approx \log f_S + \text{const}$,
   i.e. $O(\log |S|)$ — against the *diffusive linear-in-$|S|$* growth of
   $\lVert E_m \rVert^2$ under BCE. Moreover (6) makes $v^*_S$ a compromise
   across the heterogeneous user contexts $u(x)$ of *different* merged items;
   averaging conflicting directions shrinks the solution further. Empirically
   the compromise effect dominates: measured norms are flat-to-decreasing in
   $1/M$ (README, candgen finding 3).

Hence: **the $O(N/M)$ norm-growth fingerprint is a property of the loss, not
of hashing.** Practical corollary: embedding-norm monitoring as a
collision-health signal is sound for BCE ranking models and unsound for
softmax retrieval models.

## 5. Sampled softmax with logQ correction

In-batch candidates $\mathcal{B} \ni y$ with sampling distribution $Q$;
corrected logits $\tilde s_j = s_j - \log Q(j)$, batch softmax $\tilde p$ over
$\mathcal{B}$, residuals $\tilde r_j = \tilde p_j - \delta_{j,y}$. Then
$\sum_{j \in \mathcal{B}} \tilde r_j = 0$ — Lemma 1 holds *within the sampled
set*, the decomposition (2) holds with $q$ replaced by
$\tilde q = \sum_{j \in \mathcal{B}} \tilde r_j v_j$, and Lemma 4 holds for
$S \cap \mathcal{B}$. The logQ term shifts logits only — it reweights residuals
and changes no direction. All of §§2–4 apply verbatim; empirically full and
sampled agree to $\pm 0.003$ HR@10.

## 6. Normalized (cosine) towers

Let $\hat v = \tilde v / \lVert \tilde v \rVert$ and score
$s = \langle \hat u, \hat v \rangle / \tau$. For any loss,

$$\nabla_{\tilde v} L
 = \frac{1}{\lVert \tilde v \rVert}\big( I - \hat v \hat v^{\!\top} \big)\, \nabla_{\hat v} L
 \quad\Longrightarrow\quad
 \langle \tilde v, \nabla_{\tilde v} L \rangle = 0. \tag{7}$$

The radial gradient component vanishes identically: under gradient flow
$\tfrac{d}{dt}\lVert \tilde v \rVert^2 = 0$; under discrete SGD
$\lVert \tilde v' \rVert^2 = \lVert \tilde v \rVert^2 + \eta^2 \lVert \nabla \rVert^2$
— a second-order drift only. Norms are pinned by construction, and all
collision noise is carried by directions (angular dispersion). Empirically:
norms sit in a narrow band across table sizes under cosine, versus the
declining profile under dot product.

## 7. Applicability to deep sequential backbones (SASRec)

The linear towers of §1 are the objects the analysis is about. Production
retrieval, however, runs sequential encoders. SASRec holds three vocabularies
in its embedding layer and scores by a dot product:

$$\mathrm{seq}_t = E_{\text{in}}[x_t] + E_{\text{pos}}[t], \qquad
  h_t = \mathrm{Transformer}(\mathrm{seq})_t, \qquad
  s_j = \langle h_t, E_{\text{out}}[j]\rangle .$$

Under multiplexing all three share one table $E$. Which parts of §§2–4 survive?

**(a) Proposition 4.2 transfers unchanged.** The dimension-reduction result of
Coleman et al. bounds the variance of the inner-product estimator at a fixed
parameter budget; it involves neither the loss nor whatever sits between the
lookup and the score. It is therefore the *only* rigorous ground for expecting
multiplexing to beat per-vocabulary hashing here, and it is enough to motivate
the experiment.

**(b) Intra-vocabulary merging still bites, and harder.** If $j \neq j'$ collide
in the `item_out` role, then $E_{\text{out}}[j] = E_{\text{out}}[j']$ exactly,
so $s_j(x) = s_{j'}(x)$ for **every** history $x$: the two items are
permanently inseparable in the ANN index. This is the retrieval-side face of the
paper's "collisions act as value merging".

**(c) The orthogonalization mechanism does NOT transfer.** Proposition 2 needs a
*fixed* reader per vocabulary. SASRec has none:

$$\frac{\partial L}{\partial E_{\text{out}}[j]} = r_j\, h_t, \qquad
  \frac{\partial L}{\partial E_{\text{in}}[x_t]} = J_t^{\!\top}\!\left(\frac{\partial L}{\partial h}\right),$$

where $h_t$ ranges over the transformer's image and $J_t$ is its input Jacobian
— both input-dependent. The garbage directions are not confined to a fixed
subspace $\mathrm{range}(A_s^{\!\top})$, so the annihilation condition
$(\star)$ cannot even be stated. Worse, both direction sets are generically
full-rank in $\mathbb{R}^d$, and two full-rank subspaces of $\mathbb{R}^d$
cannot be mutually orthogonal (Corollary 3 would demand
$\mathrm{rank} \le d/2$ each). Coleman et al. face the same gap and argue by
analogy ("we expect deeper architectures to exhibit analogs of weight
orthogonalization due to their relative overparameterization"); we inherit it.
Consequently the SASRec arm is an **empirical** test resting on (a), not a test
of the mechanism — the mechanism is measured in the linear model, where the
readers are explicit.

**(d) Positions must never be multiplexed** — a strict corollary. Since
$\mathrm{seq}_t$ is the *sum* of the item and position rows,

$$\frac{\partial L}{\partial E_{\text{in}}[x_t]}
 \;=\; \frac{\partial L}{\partial \mathrm{seq}_t}
 \;=\; \frac{\partial L}{\partial E_{\text{pos}}[t]} \tag{8}$$

— the two roles share **one gradient direction identically**. By the taxonomy of
§2 they are not two features at all: an item/position collision is an
*intra*-feature collision (value merging), unrecoverable by any reader, with no
inter-feature component to mitigate. It is pure damage, and it buys nothing:
positions are $O(10^2)$ rows against $O(10^5..10^7)$ items, i.e. well under 1%
of the budget. `sasrec/embeddings.py` therefore keeps the position table
collisionless under every method (`NEVER_HASHED`).

The same test applies to any additive embedding composition (segment/type
embeddings in BERT-style encoders, additive context features): **if two
vocabularies are summed before any transformation, they share a gradient and
must not share a table.**

## 8. Testable predictions vs. measurements

| Prediction | Source | Status |
|---|---|---|
| Ordering NM < MX < CL at every budget; gap grows with compression | §2 (intra merging unrecoverable; MX load-balances it) | ✅ HR@10, gap $\times 2$ at $0.5\times/0.1\times$, $\gg$ 5-seed std |
| Leakage $\mathrm{ov}(A_t, A_s)$ decreases with $M$; no pressure without collisions | §3 | ✅ CL stays at identical-blocks level (0.252); MX reaches/undershoots the random baseline (0.178 < 0.183) |
| Rank collapse: $\sum_t \mathrm{rank}(A_t) + \mathrm{rank}(B) \lesssim d$ under heavy collisions | Cor. 3 | ⏳ untested |
| No $O(N/M)$ norm growth under softmax; flat-to-decreasing | §4 | ✅ (BCE ranking control: $\times 7.4$ growth) |
| Sampled+logQ $\equiv$ full softmax | §5 | ✅ $\pm 0.003$ |
| Norms pinned under cosine | §6 | ✅ band 0.42–0.46 across $20\times$ table-size range |
| Multiplexing helps a deep sequential backbone at a fixed budget | §7(a), Prop. 4.2 | ⏳ SASRec sweep running |
| Item/position collisions are pure damage (never multiplex positions) | §7(d), Eq. (8) | ✅ by derivation; enforced in code |

**Known gaps in rigor** (same level as the source paper): Proposition 2
assumes frozen readers (their footnote 2); the pressure argument in §3 and the
compromise-shrinkage argument in §4 are heuristic. For deep backbones the
readers become input-dependent Jacobians and §7(c) shows the annihilation
condition is not merely unproven but unstatable — there, only Proposition 4.2
and the intra-vocabulary merging argument carry over, and we test the rest
empirically.
