---
layout: post
title: "The Parlance of AI Safety, Ethics, and Governance"
categories: "AI"
featImg: wordsoupv2.png
excerpt: "Providing a glossary of terms to better understand the field"
permalink: "parlance-of-ai-safety"
style: 
---

---
### Introduction
Think of this post as a living collection of operational definitions. 
It is not a typology, nor a source of exacting and prescriptive terminology; rather, it is an overview of key jargon and research agendas
popular at the time of writing. I intend to edit and update periodically as the field evolves. 

[last update: 12/2024]. 

---

**AI**: In general, AI refers to the ability of a machine to perform a task typically reserved for humans. Its modern usage usually refers more explicitly to machine learning and/or reinforcement learning, with an emphasis on the capacity to learn and solve complex problems. 

**Foundational Models**: Foundational models are large, multimodal, general-purpose models trained on massive datasets. Some pieces of legislation use a compute threshold to categorize models as "foundational," but this compute-based delineation lacks consensus. Due to their scale, only a handful of companies are in the business of developing foundational models. 

**AGI**: Artifical general intelligence (AGI) is a bit of a nebulous term. It describes an AI system that performs at the level of human intelligence across all (or most) tasks. In some domains, we have arguably achieved or exceeded human level intelligence (e.g. translating languages), but full-scale, general-purpose AGI has yet to be developed. 

**Superintelligence**: Superintelligence is the next step up from AGI, describing an AI system that outperforms human intelligence. AGI/Superintelligent AI are the main source of apprehension regarding catastrophic risk from AI. 

**General Purpose Technology (GPT)**: The classic examples of GPTs[^b] are electricity and the Internet. Both of these technologies dramatically transformed society given their multifaceted and widespread usage. Historians and political scientists with expertise in historical technological innovations tend to categorize AI as a GPT, possibly with never-before-seen potential to transform society. This is precisely why it is of concern to many. 

**Frontier AI**: Frontier AI refers to future-leaning capabilities of the model(s) that are currently the most powerful---i.e. those at the frontier. Accordingly, *frontier safety* focuses on anticipating the new risks introduced by the next generation of model capabilities.   

**AI X-risk/Catastrophic Risk**: AI existential risk (X-risk) and catastrophic risk refer to the possibility that AI could cause human extinction or, at the very least, cause large-scale catastrophe. Among other examples, some hypothesize that powerful AI could lead to biological attacks, financial meltdowns, autocratic dictatorships, conflict escalation, uncontrolled climate change, and psychological manipulation, ultimately threatening the welfare of the human race. 

**Responsible AI**: Responsible AI is the umbrella under which the fields of AI safety, security, ethics, and governance fall (at least, I would argue as much). It refers to the development of increasingly capable AI without compromising human welfare in the process. 

**AI Safety**: AI Safety is usually taken to refer to technical areas of research, in some part borne out of concerns of AI X-Risk/Catastrophic Risk. One of the primary research objectives of the field is to prevent unintended negative externalities from misaligned AI systems, stemming from both accidents and malicious misuse. Key focus areas have historically included value alignment, adversarial robustness, scalable oversight, evaluations, and mechanistic interpretability, among others. 

**Adversarial robustness**: Adversarial robustness refers to the capacity of a system to maintain its performance and function in response to an adversarial attack. Usually, an adversaral attack takes the form of targeted inputs designed to manipulate or disrupt the system's intended behavior. 

**Alignment**: An aligned AI system is one whose goals and behaviors match human intentions, values, and ethics.[^c]

**Outer alignment**: When training a model, most AI paradigms require an objective function---something that tells the model if a given action is good or bad, so that it can update its parameters and try again. The first step towards achieving an aligned system is aligning the objective function. An objective function that is at odds with human intentions, welfare, or morals will naturally produce a misaligned system. This is referred to as the outer alignment problem. 

**Inner alignment**: Inner alignment is a bit trickier. For now, we do not have a perfect understanding of the internal decision-making processes of models. This means we might observe a model's behavior as achieving the intended objective, but in actuality it has learned a separate set of heuristics that could be harmful. The model's learned behavior might be correlated with the intended behavior, giving the illusion of alignment until deployment, at which point a model could misbehave; contrarily, the model's learned behavior could be completely and obviously undesirable, but still manages to optimize its objective function (e.g. via proxy gaming). 

**Explainability**: An explainable AI system is one whose decisions can be explained and understood by humans, usually via some post-hoc reasoning. An explainable system 
has more to do with the externals of a system and *why* it made a decision, but not necessarily *how*. 

**Interpretability**: Interpretability, in contrast to explainability, seeks to explain the *how* of model decision-making, emphasizing model internals. 

**Mechanistic Interpetability**: Mechanistic interpretability (aka mech interp) is 
arguably the dominant paradigm within interpretability. Mech interp seeks to reverse-engineer model internals to develop a mechanistic understanding of a model's algorithms and behavioral heuristics. 

**Evaluations**: In general, evaluations are systematic approachs to measuring the capabilities, safety, and reliability of AI systems. Researchers focused on evaluations develop benchmarks, stress-tests, alignment-tests, and performance metrics for models. Those focused on safety and capability evaluations try to develop metrics that can serve as early-warning systems for potentially harmful models. 

**Scalable Oversight**: Scalable oversight refers to the tricky problem of supervising and controlling AI systems that become increasingly powerful, complex, and intelligent. Direct human supervision is both impractical and potentially impossible for sufficiently intelligent models; scalable oversight thus seeks to develop strategies for managing systems that may become far more intelligent than any human. 

**Multi-Agent AI Systems**: These are systems wherein multiple AI agents interact with one another. Multi-agent researchers tend to focus on emergent behaviors, coordination strategies, and competitive dynamics, drawing on inspiration from game theory, evolutionary biology, autonomous systems, and economics. 

**Trustworthy AI**: We can invoke common conceptions of "trust" to inspire a definition for what constitutes trustworthy AI. Typically, to trust a person, you might want to know they are providing you with correct information; they will not cause you harm; they do not hold bias/prejudice against you; they will not disclose your sensitive information; they are held accountable for their actions; etc. The same goes for AI systems--we want trustworthy models to be accurate, harmless, unbiased, secure, and accountable. 

**Secure AI**: AI that is protected from adversarial attacks, unauthorized access, and/or malicious misuse. Secure AI usually refers to cybersecurity, model integrity, and resilience to threat actors. 

**AI Assurance**: Many of the above terms implicitly outline a wish-list for the kinds of characteristics we want intelligent systems to have. AI Assurance refers to the processes and frameworks used to actually verify that a given system has these characteristics---that it is safe, ethical, and compliant with its regulatory environment. Assurance is closely tied to auditing, risk assessment, and evaluations. 

**AI Governance**: AI Governance simply refers to the processes that determine the development, deployment, and oversight of AI. It is NOT a term used to refer exclusively to governments or to regulations, but rather encompasses the full suite of norms, policies, organizations, and regulations used to control how decisions are made and implemented. 

**AI Geopolitics**: The study of how AI influences the geopolitical world-order, power dynamics, international relations, and national security. AI Geopolitics researchers tend to focus on levers for influencing global technological competition, as well as the implications of AI on economic, military, and governmental affairs.  

**AI Ethics**: The study of moral principles and social considerations related to AI, usually as they relate to bias, fairness, accountability, privacy, societal harm, and human value alignment. Ethics is usually divided into three branches (applied, normative, and metaethics), though applied ethics is typically most relevant for AI.

[^b]: This is truly an unfortunate acronym, but one that is popularly used. It is not the same as the GPT in ChatGPT, which stands for "generative pre-trained transformer."

[^c]: This is an intentionally brief definition. Any expansion would likely become subjective or deviate from standard conceptions of what is meant by "alignment."

