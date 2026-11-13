---
layout: post
title: "Red Teaming as a Stackelberg Game"
categories: "AI Safety"
featImg: red-teaming.png
excerpt: "A mathematical framing of universal jailbreaks"
permalink: "red-teaming-stackelberg"
---

## Preliminaries

A **Stackelberg game** is a two-player sequential game where one player (the *leader*) commits to a strategy, while a second player (the *follower*) best-responds to it. For red-teaming AI models, this model maps to the following: 

- The **defender/leader** is the model developer — they commit to guardrails, fine-tuning procedures, and safety training before deployment.
- The **attacker** is the follower — they observe the deployed model and search for prompts that elicit policy-violating behavior.

Formally, let $$\mathcal{P}$$ be the space of possible prompts, and let $$f_\theta: \mathcal{P} \to \mathcal{R}$$ be the model mapping prompts to responses. The defender chooses parameters $$\theta$$ to minimize some safety loss $$\mathcal{L}_D$$; the attacker then solves:

$$p^* = \arg\max_{p \in \mathcal{P}} \mathcal{L}_A(f_\theta(p))$$

where $$\mathcal{L}_A$$ measures how policy-violating the response is. The defender's true objective is:

$$\min_\theta \mathcal{L}_D(\theta) \quad \text{subject to} \quad \mathcal{L}_A(f_\theta(p^*(\theta))) \leq \epsilon$$

The attacker's best response $$p^*$$ is itself a function of $$\theta$$, which is what turns this into a game rather than a static engineering problem. 

## Finite Red-Teaming

Standard red-teaming evaluates a fixed model against a fixed attack budget: $$n$$ human testers, $$m$$ automated attack attempts, $$T$$ hours. Generally, risk management policies (at minimum) dictate that systems must face a certain level of pre-deployment testing (subject to $$n$$, $$m$$, and $$T$$); if no attack succeeds, the model is declared fit-for-release, but this isn't a safety guarantee. It's a lower bound on attacker effort, similar conceptually to how cryptopgraphic hashes earn their "safe" bonafides. We raise the bar for a successful attack to be so high that it is unrealistic, implausible, or infeasible for an attacker to succeed given finite resources. 

To see why this works in practice, consider the size of the attack surface. The prompt space $$\mathcal{P}$$ over a vocabulary of size $$V$$ and maximum length $$L$$ has cardinality $$V^L$$. For GPT-4-level models, $$V \approx 100{,}000$$ and $$L \approx 128{,}000$$. To give you a sense of scale, the number of atoms in the observable universe is roughly $$10^{80}$$; the space of possible attacks dwarfs this number.

With more formal notation: let $$\mathcal{P}^* \subseteq \mathcal{P}$$ be the set of jailbreaking prompts. A red-team with budget $$n$$ can only certify that $$\mathcal{P}^* \cap S = \emptyset$$ for the sampled set $$S$$, where $$\lvert S \rvert = n$$. But what we are really interested in (in terms of risk management) is $$\mathcal{P}^* \setminus S$$ -- the space of unexplored potential attacks. 

So, *what is the structure of $$\mathcal{P}^*$$?* If jailbreaks are isolated points scattered through a high-dimensional space, exhaustive sampling might get you somewhere. But if they form connected manifolds/manifold-like structures (and empirical evidence strongly suggests they do), then patching one jailbreak can leave entire neighborhoods of attack types intact.

## Re-thinking Red-teaming

There's an inherent asymmetry exposed with the Stackelberg framework. The attacker needs to find *one* policy-violating prompt. The defender needs to prevent *all* of them. Equivalently, the attacker wins on a single point in $$\mathcal{P}^*$$; the defender wins only if $$\mathcal{P}^* = \emptyset$$. As a result, safety bonafides should not be seen as a binary. Instead:

**Red-teaming should be adaptive, not exhaustive.** The goal of a red-team isn't to sample broadly from $$\mathcal{P}$$. It's to simulate the attacker's best response to the *current* defense. This means iterative red-teaming where attack strategies are updated dynamically as opposed to a single fixed evaluation battery.

**Defenses should be evaluated on their effect on $$\mathcal{P}^*$$, not on pass/fail.** A defense that makes jailbreaking 1000x harder moves the needle in the right direction, even if jailbreaking remains technically possible. Current red-teaming metrics do not reflect this. 

**A Stackelberg formulation offers a sensible testing procedure** If the attacker always best-responds to the deployed model, the defender's training should include that best response. This is exactly what adversarial fine-tuning and tamper-resistant training attempt (i.e. simulating the attacker's move during training, not after deployment).
