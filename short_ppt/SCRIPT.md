# Talk script — Linear ICA via Optimal Transport

Deliberately long (~20-22 min at a natural pace, ~3000 words). Cut freely —
suggested trims are marked **[CUT CANDIDATE]**. Each section header names the
slide so you can match script to deck. Bracketed stage directions tell you
when to let the animation play before talking over it.

Rough budget if you keep everything: title 1.0, theorem1 1.0, whitening 1.5,
cardoso 1.5, OT basics 1.5, Wasserstein-as-contrast 1.5, theorem3 1.5,
mixing 1.5, algorithms table 1.0, contrast-race 1.5, comparison 1.5,
optimization-w2 1.5, optimization-practical 1.0, EEG 1.5, limitations 1.5,
close 0.5 ≈ 21 minutes.

---

## 1. Title

Good [morning/afternoon], everyone. I'm going to talk about a question that
sits right at the intersection of two fields that don't normally talk to
each other: Independent Component Analysis, and Optimal Transport theory.

Here's the setup in one sentence. Independent Component Analysis — ICA — is
the problem of recovering a set of mutually independent source signals from
only their linear mixture: you observe the mixed signal, you don't observe
the sources, and you don't know the mixing matrix either. Optimal Transport,
on the other hand, is a branch of probability theory that gives us an actual
*metric* — the Wasserstein distance — for how far apart two probability
distributions are, in the sense of the cheapest way to morph one into the
other.

The question this work asks is simple to state and, it turns out, not at all
simple to answer: **can we use that Wasserstein metric as the contrast
function that drives ICA?** That's the thread I'll follow for the next
twenty minutes — what the classical contrast functions look like, why they
struggle, what Optimal Transport offers instead, and where it still falls
short.

---

## 2. The problem: linear ICA

Let's ground this properly. The ICA model says: we observe $\mathbf{X}$,
which is an invertible linear mixture $\mathbf{A}$ of latent sources
$\mathbf{S}$. The sources are mutually independent, and — this is the
condition that does all the work — at most one of them is allowed to be
Gaussian.

Under those three assumptions, the mixing matrix $\mathbf{A}$ is
identifiable up to two harmless ambiguities: you can't recover the *order*
of the sources, and you can't recover their *scale or sign*. Those are
baked into the problem itself — if I hand you a mixture, there's no way to
know whether source 1 was originally twice as loud as I'm showing you, or
whether I've swapped source 1 and source 3. Fine. Everything else, though,
is recoverable.

**[CUT CANDIDATE — the rotation-invariance explanation can be shortened to one sentence if time is tight]**

Now, why the Gaussian condition? Here's the intuition. Any linear combination
of Gaussian random variables is again Gaussian. And the multivariate
Gaussian distribution has a very inconvenient property for us: it's
rotationally invariant — it looks exactly the same from every angle once
you've decorrelated it. So if two or more of your sources are Gaussian,
there is no statistical handle left to tell rotations apart, and the
mixing matrix becomes fundamentally unrecoverable beyond decorrelation. The
flip side is the promise of this whole talk: as soon as you have two or more
*non*-Gaussian sources, there genuinely is a unique direction to search for.
The rest of this talk is about how to find it.

---

## 3. Centering, whitening, and the search space

[Let the animation run for a few seconds before narrating.]

So how do we actually go about finding that direction? The standard
pipeline has two preprocessing steps. First, **centering** — just subtract
the mean. Second, **whitening** — apply a linear transform so that the
covariance becomes the identity matrix.

What you're watching here is exactly that, geometrically. We start with a
raw mixture of two independent Laplace sources — that's the skewed,
correlated-looking cloud. We center it, so it moves to sit on the origin.
Then we whiten it, and the cloud reshapes into something with unit
covariance in every direction — no more linear correlation left to exploit.

Here's the subtlety I want you to sit with for a second: notice that the
whitened cloud is *not* circular. It's still got a clear diamond, almost
star-like shape to it. That's the whole point of this slide. Whitening kills
*linear* correlation — second-order structure — but it does nothing to the
higher-order, genuinely statistical structure of the sources. There is
still a rotation hiding in this picture, and the circle you see appearing is
exactly the space we now have to search: the unit sphere of candidate
directions $\mathbf{b}$, also called the orthogonal group.

Formally, our goal becomes: find $\hat{\mathbf{b}}$ that maximizes some
**Contrast** function of the projected data. I've deliberately left
"Contrast" undefined on this slide. It's some measure of independence — and
figuring out exactly what it should be is the rest of this talk.

---

## 4. The Cardoso idea: independence as non-Gaussianity

This next slide is, for me, the single most elegant piece of the theoretical
story, and it comes from Jean-François Cardoso's information-geometry
framing of ICA.

Picture the space of all possible joint distributions. Within that space,
there are two special subsets: the **Product manifold**, containing every
distribution whose marginals are independent — that's literally where we
want to end up — and the **Gaussian manifold**, containing every possible
multivariate Gaussian. Cardoso's trick is to project our actual joint
distribution $P_Y$ onto *both* of these subsets and measure the KL-divergence
each way.

[Point at the figure.] What you're looking at is two right triangles sharing
a hypotenuse. The blue triangle is the path through the Product manifold:
the divergence from $P_Y$ to the point representing "fully decorrelated and
factorized" splits into the mutual information $\mathcal{I}(Y)$ — how far
$P_Y$ is from being a product distribution — plus the sum of marginal
non-Gaussianities. The orange triangle is the path through the Gaussian
manifold: the same total divergence splits instead into the joint
non-Gaussianity $G(Y)$ plus a correlation term $C(Y)$. Because both paths
measure the exact same quantity, we can set them equal — that's the
"same LHS" step in the figure — and that gives us Cardoso's unified theorem.

Here's the payoff. Once we whiten our data, the covariance is identity, so
that correlation term $C(Y)$ drops to exactly zero. The identity collapses
to: mutual information equals joint non-Gaussianity minus the sum of
marginal non-Gaussianities. And because the joint non-Gaussianity term
$G(Y)$ doesn't change under rotation — it's rotation-invariant — minimizing
mutual information, i.e. finding independence, becomes *exactly equivalent*
to maximizing the sum of the marginal non-Gaussianities.

So: independence, the thing we actually want, has just turned into
non-Gaussianity, a thing we might know how to measure. Great — now we need
an actual contrast function for non-Gaussianity. Where do we find one?

---

## 5. Optimal Transport: the basics

This is where Optimal Transport enters the story, so let me give you the
thirty-second version of OT theory.

The original question, due to Monge in 1781, was beautifully physical: you
have a pile of mass, you have a target shape, what's the cheapest way to
move every grain of the pile onto the target? Monge imagined this as a
deterministic map $T$ — every point goes to exactly one destination.

**[CUT CANDIDATE — the next two sentences on why push-forward maps aren't always enough can be trimmed]**

The trouble is that a single deterministic map doesn't always exist — if
you're moving between two discrete distributions with mismatched point
masses, there's simply no way to send each source point to exactly one
target point and balance the books. Kantorovich's fix, in 1942, was to
relax this: instead of insisting on a single map, allow mass to *split*
across what's called a transport plan, or coupling — a joint distribution
whose two marginals are exactly your source and target. A coupling always
exists, and a push-forward map is just the special, lucky case where that
coupling happens to be deterministic.

That's the move that lets us define the Wasserstein distance: it's the cost
of the *cheapest* transport plan, the minimum, over every valid coupling, of
the expected transport cost. [Gesture at the formula.] And critically,
this satisfies the triangle inequality — it's a genuine metric on the space
of probability distributions, not just a divergence or a similarity score.
That distinction matters a lot for what's coming next.

---

## 6. Wasserstein as contrast

Now, why specifically the *squared* Wasserstein-2 distance, and not, say,
Wasserstein-1?

The answer is a regularity result due to Caffarelli: for the quadratic
cost specifically, the optimal transport map is guaranteed to be smooth.
That smoothness is exactly what we need for a contrast function we're going
to optimize with gradients — $W_1$ and other choices of $p$ don't come with
that same guarantee, and empirically their optimization landscapes are much
more jagged.

In one dimension, Brenier's theorem plus that same Caffarelli regularity
tells us something very concrete: the optimal map between any two
1D distributions is *monotone* — it preserves order. And that's a huge
computational gift, because it means $W_2^2$ reduces to a simple
quantile-matching integral.

**[CUT CANDIDATE — the rank/sorting explanation is the most "textbook" part of the talk; can compress to two sentences]**

Here's why monotonicity is such a gift, concretely: if the map preserves
order, it has to send the $k$-th smallest sample of your data to the $k$-th
smallest value of the target. That means the entire map is just: sort your
data, get its rank, divide by $N$ to get the empirical CDF, then look up the
matching quantile on the target side. At no point do we ever evaluate a
*density* — there's no kernel density estimate, no bandwidth parameter to
tune, just sorting. That's a sharp departure from classical contrast
functions, which almost always smuggle in some density approximation.

So here's our actual contrast: the squared Wasserstein-2 distance from the
projected data to a standard Gaussian $\Gamma$. The generic "Contrast" from
three slides ago finally becomes concrete — we maximize $W_2^2$ over all
unit-norm directions $\mathbf{b}$ on the orthogonal group.

---

## 7. Why maximizing $W_2^2$ recovers sources

Of course, stating an objective isn't the same as proving it works. This
slide is the theoretical heart of the contribution.

The first theorem says: for any unit-norm mixture of mutually independent
sources, the squared Wasserstein distance of the *mixture* to a Gaussian is
upper-bounded by the weighted sum of the squared Wasserstein distances of
the individual *sources* to that same Gaussian. In plain language: mixtures
are, in this metric, never more non-Gaussian than a weighted combination of
their parts — which should feel familiar, it's a Central-Limit-Theorem-style
statement.

The proof idea is a coupling trick: take a single shared noise vector,
push it through the optimal transport map for each source separately to
build the mixture side, and push it through the identity to build the
Gaussian side. Because the two constructions share the same randomness, the
cross terms in the resulting expectation cancel by independence, and what's
left is exactly the bound. Full algebra is in the appendix if anyone wants
it.

The second theorem upgrades this to a *strict* inequality whenever at least
two weights are non-zero and at most one source is Gaussian — equality
would require the optimal transport maps to be linear, which would make the
sources Gaussian, contradicting our assumption. Put the two together and you
get the result we actually need: the global maximum of $W_2^2$ over the
whole orthogonal group is attained **only** when $\mathbf{b}$ aligns with a
single pure source direction. That's the theoretical license to go
maximize this thing and trust the answer.

---

## 8. Seeing it happen: mixing two sources

[Let the animation play through at least one full sweep before talking.]

Theory's nice, but let's actually watch this happen. Here are two unit
variance sources: a Laplace, which is peaked and heavy-tailed, and a
Uniform, which is flat and bounded. I'm mixing them with a single angle
$\theta$ — at $\theta=0$ you get pure Laplace, at $\theta=90°$ you get pure
Uniform, and in between you get a genuine two-source mixture.

Watch the histogram against the dashed Gaussian curve as $\theta$ sweeps.
At the endpoints, the histogram is visibly *not* Gaussian — Laplace has
that sharp peak, Uniform is flat-topped. But right in the middle, around
$\theta=45°$, the mixture hugs the Gaussian curve almost exactly. That's the
Central Limit Theorem made visible: a mixture of independent non-Gaussian
sources looks more Gaussian than either source alone.

And the curve on the right is tracking exactly our contrast, $W_2^2$ to a
standard Gaussian, as a function of that same angle. It dips right where
the histogram looks most Gaussian, and it peaks at the two pure sources.
That dip-and-peak shape is precisely the signal an ICA solver climbs to find
the true directions.

---

## 9. Classical ICA: same goal, different contrasts

So OT-ICA isn't proposing a new *goal* — every classical algorithm is
chasing the exact same non-Gaussianity-maximization idea. They just differ
in *how* they approximate it.

FastICA uses a logcosh approximation to negentropy — a smooth proxy that's
cheap to differentiate. JADE instead diagonalizes fourth-order cumulant
matrices jointly. InfoMax maximizes a parametric log-likelihood under a
fixed logistic density assumption. Picard keeps that same likelihood but
replaces the solver with something that provably converges faster.

**[CUT CANDIDATE — this whole slide can be summarized in two sentences if you're behind schedule]**

What unites the first four rows is that every single one of them replaces
the *true* non-Gaussianity with something cheaper to estimate from
low-order statistics, or under a fixed density assumption. And each of
those proxies has specific source geometries where it breaks down — that's
exactly what we're about to see.

---

## 10. What each contrast function actually does

[Let this animation run through a full sweep — it's the centerpiece, give it room.]

This is, I think, the most direct way to *see* what "maximizing a contrast
function" actually means in practice. I've taken a simple two-source toy
mixture — Laplace and Uniform again, but now already expressed in
source-aligned coordinates, so the true answer is exactly 0° and 90°.

On the left, a candidate direction $\mathbf{b}(\theta)$ sweeps around the
unit circle. On the right, all four contrast functions — FastICA, JADE,
InfoMax, and our $W_2^2$ — are evaluated at that same angle, each
normalized so we can compare *where* they peak rather than their raw scale.

Watch them rotate together. All four land their maximum on the same two
independent candidates — that's reassuring, it confirms they're all
genuinely chasing non-Gaussianity. But look at the *sharpness* of each
peak: that differs quite a bit between methods, and that difference in
sharpness is exactly what turns into a difference in statistical efficiency
and robustness once you move to harder, higher-dimensional, more
heterogeneous mixtures — which is exactly the next slide.

---

## 11. Performance at high dimensions, heterogeneous mixtures

So does that sharper or smoother peak actually translate into better
recovery? We ran a proper ablation to check: dimensions 10, 20, and 30,
ten thousand samples, ten trials, across five different mixture regimes
ranging from purely continuous to a full hybrid of continuous, discrete,
and Gaussian sources together.

The dashed line marks an Amari error of 0.3 — a standard threshold for
"good separation" in the ICA literature; full definition is in the
appendix if you want the exact heuristic scale.

Here's the headline: as dimension grows and the sources get more
heterogeneous, OT-ICA's advantage over the proxy methods actually *grows*
with it. On purely continuous sources, we're 40 to 45 percent lower error
than FastICA. On the full hybrid mixture — continuous, discrete, and
Gaussian sources all thrown in together — that gap widens to a factor of
2 to 5 times.

**[CUT CANDIDATE — the discrete-only nuance is honest but can be dropped for time]**

The one place this doesn't hold is purely discrete sources: JADE actually
wins at the smallest dimension, $d=10$. But by $d=20$, OT-ICA takes the lead,
and at $d=30$ it's the *only* method that doesn't saturate the error
ceiling completely. Full numbers for every regime and every method are in
the appendix table.

---

## 12. Optimization with $W_2^2$

Let's switch from "does this work" to "how do you actually make it work,"
because maximizing $W_2^2$ over a single direction only ever gives you the
*single* most non-Gaussian direction. Recovering all $d$ sources needs more
machinery.

The natural idea is **iterative deflation**: find the first direction
$\mathbf{b}_1$, then restrict the next search to whatever is orthogonal to
it, find $\mathbf{b}_2$, restrict again, and so on. And this isn't just a
convenient heuristic — it's actually *provably* lossless here, because the
unmixing matrix is constrained to the orthogonal group in the first place,
which means the true recovery directions are already mutually orthogonal.
Restricting the search to the orthogonal complement of what you've already
found can never accidentally exclude a true remaining source.

**[CUT CANDIDATE — if time is short, you can skip straight from "iterative deflation" to the joint-symmetric fix and just say "naive deflation has an error-accumulation problem, so instead we..."]**

But naive sequential deflation has a real practical flaw: any small error
in $\mathbf{b}_1$ propagates into the orthogonal complement you search for
$\mathbf{b}_2$, which propagates into the complement for $\mathbf{b}_3$, and
so on — errors compound down the chain.

The fix is to stop treating this sequentially at all. Instead, treat the
*entire* unmixing matrix $\mathbf{B}$ as a single point on the orthogonal
group, and update all $d$ rows jointly. Every optimization step retracts
the whole matrix at once via a symmetric decorrelation — that's the
$(\mathbf{B}\mathbf{B}^\top)^{-1/2}\mathbf{B}$ formula — which spreads any
orthogonality violation evenly across every component, rather than treating
the earlier-found ones as ground truth and forcing the rest to adjust around
them, the way something like Gram-Schmidt would.

---

## 13. Making $W_2^2$ optimization practical

**[CUT CANDIDATE — this whole slide can become a single sentence: "we use a handful of standard manifold-optimization tricks to make this fast — sorting tricks, dithering for discrete data, Riemannian gradients, and L-BFGS"]**

Putting that into an actual working algorithm needs a handful of supporting
techniques, and I've laid them out here in the order they actually fire
during one optimization step.

We precompute analytical Gaussian targets once up front — the exact
expectation within each quantile bin, rather than a sampled approximation,
which removes one whole source of gradient noise. Each step, we draw a
batch — parallelizing many random restarts at once and subsampling the
data, which helps escape spurious local optima, especially on discrete
mixtures. We inject a small amount of Gaussian dithering before sorting,
which smooths the step-function CDFs that discrete sources produce into
differentiable curves. We compute the Euclidean gradient and project it
onto the tangent space of the orthogonal group — that's the Riemannian
gradient step. We refine that step using L-BFGS, a quasi-Newton method,
because the *exact* Newton Hessian here would require a density estimate
right at the quantile boundary, which is the one place this sorting-based
approach can't avoid density estimation — so we approximate curvature from
gradient history instead. And finally we retract back onto the orthogonal
group via that same symmetric decorrelation from the last slide.

---

## 14. Application: EEG ocular-artifact removal

Let's ground all of this in a real application. The skull behaves as a
linear volume conductor, so an eye blink and the underlying neural signal
mix linearly by the time they reach a scalp electrode — this is, structurally,
exactly the ICA problem.

[Let the scrolling EEG animation run for a few seconds.]

We took five frontal channels from the MNE sample dataset and ran OT-ICA.
It isolates the blink into a single component, identified purely by its
excess kurtosis — about 60 for that one component versus essentially zero
for the rest — with no density model assumed anywhere.

Quickly, what RMS means here: it's the root-mean-square amplitude in a time
window, essentially the local signal power. A blink injects a large
transient that inflates RMS sharply in that window. Zeroing the isolated
artifact component and reconstructing gives us better than 90 percent RMS
reduction in that blink window, while leaving the rest of the recording
completely untouched.

**[CUT CANDIDATE — the downstream-significance sentence is good context but skippable]**

Why does this matter practically? Downstream EEG analysis — event-related
potential studies, brain-computer interfaces, clinical reading — needs
artifact-free signal, and the usual alternative is to just throw away any
trial contaminated by a blink. This removes the artifact instead of the
data.

---

## 15. Limitations and what's next

Let me close with where this stands and where it doesn't.

The case for OT-ICA, in one sentence: $W_2^2$ is a genuine metric on
probability distributions, not a moment or likelihood proxy, and that's
exactly why it holds up better across heterogeneous mixtures where the
proxies are misspecified for at least one source type.

Two honest limitations. First, discrete sources: step-function CDFs create
gradient plateaus that stall the solver, even though the contrast signal
itself is genuinely present — it's an optimization-landscape problem, not a
contrast problem. Gaussian dithering smooths this partially; entropic
Sinkhorn distances might smooth the transport plan further still.

Second, speed: per-iteration cost scales as $K$ times $d N \log N$, against
FastICA's $dN$ — sorting is just more expensive than a fixed-point update.
**[CUT CANDIDATE — the amortized-Sinkhorn explanation is somewhat speculative future work; can be shortened to "amortized Sinkhorn solvers may help close this gap"]**
*Amortized* Sinkhorn distances might help close that gap: instead of solving
the entropic transport problem completely from scratch at every single
step, you warm-start the dual potentials from the previous step's nearby
rotation — since consecutive optimization steps query very similar
projections, the iterative cost gets spread out across the whole
trajectory instead of being paid in full every time.

And looking forward: we see OT-ICA as a natural front-end for LiNGAM-style
causal discovery, for Causal Component Analysis, and for identifiable
nonlinear ICA — all three of those methods need exactly the kind of clean,
distribution-free independence signal that this contrast provides.

That's the talk. Happy to go deeper into any of the proofs, the full
results table, or the references — they're all in the backup slides. Thank
you.

---

## Appendix slides — what to say if asked

- **A1 / A2 (proofs):** "Here's the full coupling argument behind the bound,
  and here's how the strict version builds directly on it."
- **A3 (Amari index):** "This is the exact metric and the heuristic
  thresholds I referenced on the comparison slide."
- **A4 (full table):** "Every number from the ablation, regime by regime,
  dimension by dimension."
- **A5 (references):** self-explanatory if someone asks for a specific
  citation.
