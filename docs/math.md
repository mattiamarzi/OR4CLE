# Mathematical notes for OR4CLE

This document summarizes the Bayesian predictive framework implemented
in OR4CLE.

We consider undirected binary networks. At time $t$ we observe an
adjacency matrix $A(t)$ with entries

$$
A_{ij}(t) \in \{0,1\}, \quad A_{ij}(t)=A_{ji}(t), \quad A_{ii}(t)=0.
$$

The degree of node $i$ is

$$
k_i(t) = \sum_{j \ne i} A_{ij}(t),
$$

and the number of links is

$$
L(t) = \sum_{i<j} A_{ij}(t).
$$

OR4CLE implements two Bayesian predictive models:

-   **BFM** (Bayesian Fitness Model),
-   **BERM** (Bayesian Erdős--Rényi Model).

------------------------------------------------------------------------

# 1. Bayesian Fitness Model (BFM)

## Likelihood

The density-corrected Gravity Model (dcGM) defines

$$
p_{ij}(z,t) = \frac{z\, s_i(t) s_j(t)}{1 + z\, s_i(t) s_j(t)},
$$

where $s_i(t)$ are observed strengths and $z>0$ is a global density
parameter.

The likelihood reads

$$
P(A(t)\mid z)
=
\prod_{i<j}
\left(
p_{ij}(z,t)
\right)^{A_{ij}(t)}
\left(
1 - p_{ij}(z,t)
\right)^{1-A_{ij}(t)}.
$$

Using the dcGM form,

$$
P(A(t)\mid z)
=
\prod_{i<j}
\frac{(z s_i(t)s_j(t))^{A_{ij}(t)}}
{1 + z s_i(t)s_j(t)}.
$$

------------------------------------------------------------------------

## Posterior distribution

Given a prior $\pi(z)$,

$$
P(z \mid A(t))
=
\frac{P(A(t)\mid z)\,\pi(z)}
{\int_0^{\infty} P(A(t)\mid z)\,\pi(z)\,dz}.
$$

In log-space with $u=\log z$, the log-posterior becomes

$$
g(u)
=
L(t)\,u
-
\sum_{i<j}
\log\bigl(1 + e^{u}s_i(t)s_j(t)\bigr)
+
\log \pi(e^{u})
+
u.
$$

------------------------------------------------------------------------

## Posterior predictive probability

The predictive distribution for the next snapshot is

$$
P(A(t+1)\mid A(t))
=
\int_0^{\infty}
P(A(t+1)\mid z)\,
P(z\mid A(t))\,dz.
$$

The marginal predictive edge probability is

$$
q_{ij}(t+1)
=
\int_0^{\infty}
\frac{z\, s_i(t+1)s_j(t+1)}
{1 + z\, s_i(t+1)s_j(t+1)}
\,P(z\mid A(t))\,dz.
$$

Numerically,

$$
q_{ij}(t+1)
\approx
\frac{1}{M}
\sum_{m=1}^M
\frac{z^{(m)} s_i(t+1)s_j(t+1)}
{1 + z^{(m)} s_i(t+1)s_j(t+1)}.
$$

Expected observables:

$$
\langle L(t+1)\rangle
=
\sum_{i<j} q_{ij}(t+1),
$$

$$
\langle k_i(t+1)\rangle
=
\sum_{j\ne i} q_{ij}(t+1).
$$

------------------------------------------------------------------------

# 2. Bayesian Erdős--Rényi Model (BERM)

## Likelihood

$$
P(A(t)\mid p)
=
p^{L(t)} (1-p)^{V-L(t)},
$$

with

$$
V = \frac{N(t)(N(t)-1)}{2}.
$$

------------------------------------------------------------------------

## Conjugate Beta prior

Assume

$$
\pi(p)
=
\frac{p^{\alpha-1}(1-p)^{\beta-1}}
{B(\alpha,\beta)}.
$$

Then

$$
P(p\mid A(t))
=
\text{Beta}\bigl(L(t)+\alpha,\; V-L(t)+\beta\bigr).
$$

------------------------------------------------------------------------

## Posterior predictive

$$
q(t+1)
=
\int_0^1
p\, P(p\mid A(t))\, dp
=
\frac{L(t)+\alpha}
{V+\alpha+\beta}.
$$

Thus

$$
q_{ij}(t+1)
=
\frac{L(t)+\alpha}
{V+\alpha+\beta}.
$$

The predictive link count follows

$$
L(t+1)\mid A(t)
\sim
\text{BetaBin}\bigl(V,\; L(t)+\alpha,\; V-L(t)+\beta\bigr).
$$
