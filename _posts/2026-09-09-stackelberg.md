---
layout: post
title: "Red Teaming as a Stackelberg Game"
categories: "AI Safety"
featImg: red-teaming.png
excerpt: "A mathematical framing of universal jailbreaks"
permalink: "red-teaming-stackelberg"
---

## The Setup

A **Stackelberg game** is a two-player sequential game where one player (the *leader*) commits to a strategy first, and a second player (the *follower*) best-responds to it. The red-teaming problem fits this mold almost perfectly.

- The **defender** (model developer) is the leader — they commit to guardrails, fine-tuning procedures, and safety training before deployment.
- The **attacker** is the follower — they observe the deployed model and search for prompts that elicit policy-violating behavior.

Crucially, the attacker moves second with (conditional) knowledge of the deployed system.

Formally, let $$\mathcal{P}$$ be the space of possible prompts, and let $$f_\theta: \mathcal{P} \to \mathcal{R}$$ be the model mapping prompts to responses. The defender chooses parameters $$\theta$$ to minimize some safety loss $$\mathcal{L}_D$$; the attacker then solves:

$$p^* = \arg\max_{p \in \mathcal{P}} \mathcal{L}_A(f_\theta(p))$$

where $$\mathcal{L}_A$$ measures how policy-violating the response is. The defender's true objective is:

$$\min_\theta \mathcal{L}_D(\theta) \quad \text{subject to} \quad \mathcal{L}_A(f_\theta(p^*(\theta))) \leq \epsilon$$

This constraint is where things become diffiuclt; the attacker's best response $$p^*$$ is itself a function of $$\theta$$. Defending against a static attack distribution is an engineering problem, while defending against an adversary who adapts to your defense is a game.

<!-- ![Stackelberg game tree: defender chooses θ at root; only the realized θ's attacker subtree is active, branching over an effectively unbounded prompt space (shown with ellipses) toward the highlighted terminal node p*](../assets/img/blog/red-teaming-game-tree.png){:class="pattern-examples"} -->

## Why Finite Red-Teaming Can't Certify Safety

Standard red-teaming evaluates a fixed model against a fixed attack budget: $$n$$ human testers, $$m$$ automated attack attempts, $$T$$ hours. Generally, risk management policies (at minimum) dictate that systems must face a certain level of pre-deployment testing (subject to $$n$$, $$m$$, and $$T$$); if no attack succeeds, the model is declared fit-for-release. Crucially, this isn't a safety guarantee. It's a lower bound on attacker effort, similar conceptually to how cryptopgraphic hashes earn their "safe" bonafides. 

To see why this works in practice, consider the size of the attack surface. The prompt space $$\mathcal{P}$$ over a vocabulary of size $$V$$ and maximum length $$L$$ has cardinality $$V^L$$. As a real-world example, for GPT-4-level models, $$V \approx 100{,}000$$ and $$L \approx 128{,}000$$. "Astronomically large" is quite literally inadequate--the number of atoms in the observable universe is roughly $$10^{80}$$, and this space of attacks dwarfs it.

More precisely: let $$\mathcal{P}^* \subseteq \mathcal{P}$$ be the set of jailbreaking prompts. A red-team with budget $$n$$ can only certify that $$\mathcal{P}^* \cap S = \emptyset$$ for the sampled set $$S$$, where $$\lvert S \rvert = n$$. That says nothing about $$\mathcal{P}^* \setminus S$$.

The question that actually matters: *what is the structure of $$\mathcal{P}^*$$?* If jailbreaks are isolated points scattered through a high-dimensional space, exhaustive sampling might get you somewhere. But if they form connected manifolds — and empirical evidence strongly suggests they do — then patching one jailbreak leaves an entire neighborhood intact.
<!-- 
![Schematic of prompt space P as a 2D projection. Safe region shaded blue, jailbreak manifold P* in orange. Two rounds of red-team samples (dots, diamonds) both miss the manifold entirely.](../assets/img/blog/red-teaming-manifold.png){:class="pattern-examples"} -->

## The Defender's Dilemma

The Stackelberg formulation exposes an asymmetry that finite red-teaming cannot paper over.

The attacker needs to find *one* policy-violating prompt. The defender needs to prevent *all* of them. In security terms: the attacker wins on a single point in $$\mathcal{P}^*$$; the defender wins only if $$\mathcal{P}^* = \emptyset$$. These are not symmetric burdens. The attacker's problem is a search problem. The defender's is a coverage problem over an infinite space.

This asymmetry makes the minimax problem structurally grim. Define the defender's minimax safety loss:

$$V^* = \min_\theta \max_{p \in \mathcal{P}} \mathcal{L}_A(f_\theta(p))$$

For current LLMs, there is substantial empirical evidence that $$V^* > 0$$ — that is, for any $$\theta$$, some adversarial prompt exists.[^3] If true, the Stackelberg equilibrium involves an attacker who always wins eventually. The question shifts from *can we prevent all attacks* to *how much effort should an attacker expend*.

That reframing is actually useful. Rather than binary safety, it motivates a cost-based analysis: what is the minimum attacker effort $$n^*(\epsilon)$$ required to find a prompt achieving loss $$\geq \epsilon$$? A model that requires $$n^* = 10^{12}$$ queries to jailbreak is meaningfully safer than one requiring $$n^* = 10^3$$, even if neither is certifiably secure. The gap between those two numbers can be the gap between a theoretical vulnerability and a practical one.

<!-- ![Plot of minimum attacker effort n*(ε) vs. violation threshold ε for three defense regimes. Higher and further right is better. Adversarial training (TAR) shifts the curve most dramatically.](../assets/img/blog/red-teaming-cost-curve.png){:class="pattern-examples"} -->

## What This Changes

If we accept the Stackelberg framing, a few things follow.

**Red-teaming should be adaptive, not exhaustive.** The goal of a red-team isn't to sample broadly from $$\mathcal{P}$$ — that's hopeless. It's to simulate the attacker's best response to the *current* defense. That means iterative red-teaming where attack strategies are updated based on what fails, not a single fixed evaluation battery run before launch and never revisited.

**Defenses should be evaluated on their effect on $$n^*$$, not on pass/fail.** A defense that makes jailbreaking 1000x harder is real progress, even if jailbreaking remains technically possible. The field needs metrics that reflect this. Pass/fail invites a false precision: "no attacks found in 10,000 tries" sounds rigorous and isn't.

**The Stackelberg structure tells you what training loop to run.** If the attacker always best-responds to the deployed model, the defender's training should include that best response. This is exactly what adversarial fine-tuning and tamper-resistant training[^4] attempt — simulate the attacker's move during training, not after deployment.

None of this makes the problem tractable. Solving the minimax problem exactly is computationally out of reach for the same reasons that make the attack space vast. But framing the problem correctly is a prerequisite for making progress. Expecting a finite red-team to certify safety isn't just optimistic — it's asking the wrong question entirely.

The right question is how much it costs to find a violation, and how much that number shifts with the release of increasingly sophisticated dual-use models. 

---

[^1]: This is a simplified version of the "DAN" prompt family that circulated on Reddit in late 2022. The specific compliance behavior varied by model and version.
[^3]: This is not proven formally for current models but is widely assumed in the security community based on empirical evidence. A formal proof would require characterizing the capacity of fine-tuning to eliminate all policy-violating behaviors, which remains open.
