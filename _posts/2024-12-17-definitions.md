---
layout: post
title: "The Landscape of Responsible AI"
categories: "AI"
featImg: neuralnetwork.png
excerpt: "Defining terms to better understand the field"
permalink: "landscape-of-responsible-ai"
style: 
---

---
### Introduction
This post surveys the field of Responsible AI, clarifying terminology and providing context and motivations for subfields. 
It is not a typology, nor a source of exacting definitions; rather, it is an overview of the key concepts and research agendas
popular at the time of writing 

[last update: 12/2024]. 

---

**AI**: In general, AI refers to the ability of a machine to perform a task typically reserved for humans. Its modern usage usually refers more explicitly to machine learning and/or reinforcement learning, with an emphasis on the capacity to learn and solve complex problems. 

**Foundational Models**: Foundational models are large, multimodal, general-purpose models trained on massive datasets. Some pieces of legislation use a compute threshold to categorize models as "foundational," but this compute-based delineation lacks consensus. Because of their scale, only a handful of companies are in the business of developing foundational models. 

**AGI**: Artifical general intelligence (AGI) a bit of a nebulous term. It describes an AI system that performs at the level of human intelligence across all (or most) tasks. In some domains, we have arguably achieved or exceeded human level intelligence (e.g. translating languages), but full-scale, general-purpose AGI has yet to be developed. 

**Superintelligence**: Superintelligence is the next step up from AGI, describing an AI system that outperforms human intelligence. AGI/Superintelligent AI are the main source of apprehension regarding catastrophic risk from AI. 

**Frontier AI**: Frontier AI refers to future-learning capabilities of the model(s) that are currently the most powerful--i.e. those at the frontier. Accordingly, *frontier safety* focuses on anticipating the new risks introduced by the next generation of model capabilities.   

**AI X-risk/Catastrophic Risk**: AI existential risk (X-risk) and catastrophic risk refer to the possibility that AI could cause human extinction or, at the very least, cause large-scale catastrophe. Among other examples, some hypothesize that powerful AI could lead to biological attacks, financial meltdowns, autocratic dictatorships, conflict escalation, uncontrolled climate change, and psychological manipulation, threatening the prosperity of the human race writ large. 

**Responsible AI**: In my view, responsible AI is the umbrella under which many subfields of AI safety, security, ethics, and governance fall. Responsible AI refers to the responsible development of increasingly capable systems without compromising safety, ethics, or human welfare in the process. 

**AI Safety**: AI Safety is a fairly specialized area of research, somewhat borne out of concerns of AI X-Risk/Catastrophic Risk. One of the primary research objectives of the field is to prevent unintended negative externalities from misaligned AI systems, stemming from both accidents and malicious misuse. Key focus areas include value alignment, adversarial robustness, scalable oversight, eavluations, and mechanistic interpretability among others. 

**Adversarial robustness**: Adversarial robustness refers to the capacity of a system to maintain its performance and function in response to an adversarial attack. Usually, an adversaral attack takes the form of targeted inputs designed to manipulate or disrupt the system's intended behavior. 

**Alignment**: An aligned AI system is one whose goals and behaviors match human intentions, values, and ethics. Given the breadth and mutual exclusivity of value systems around the world, the alignment problem is an incredibly daunting task. 

**Outer alignment**: When training a model, most AI paradigms require an objective function--something that tells the model if a given action is good or bad, so that it can update its parameters and try again. The first step towards achieving an aligned system is aligning the objective function. An objective function that is at odds with human intentions, welfare, or ethics will naturally produce a misaligned system. This is referred to as the outer alignment problem. 

**Inner alignment**: Inner alignment is a bit trickier. At least for now, we do not have a perfect understanding of the internal decision-making processes of models. This means we might observe a model's behavior as achieving the intended objective, but in actuality it has learned a separate set of heuristics that could be harmful. For example, let's say we are training an AI system to INSERT EXAMPLE IN HERE LATER

**Explainability**: 

**Interpretability**
**Mechanistic Interpetability**
**Evaluations**
**Scalable Oversight**
**Multi-Agent Systems**

**Trust and Safety**: In line with traditional Trust and Safety teams in the tech-world, AI Trust and Safety focuses on short-term accidents and misuse. Making sure models don't produce harmful content, keeping user information private and secure, staying ahead of threat actors/adversaries--these are the types of issues that fall under the domain of AI Trust and Safety. 

**Trustworthy and Secure AI**
**Trustworthy AI**
**Secure AI**
**AI Assurance**

**AI Governance**
**AI Geopolitics**
**General Purpose Technology (GPT)**
**AI Ethics**

[^a]: As a fun sidenote, a few researchers have even explored taste/smell as potential sources of inspiration for AI development. 


Until relatively recently, most work in AI could be distinctly categorized as speech/language processing, computer vision, or robotics. These fields arguably emulate the main ways in which humans interact with the world--through sight (computer vision), sound (speech/language processing), and physical touch (robotics)[^a]. Crucially, early models in these fields were quite small and specialized in comparison to today's models. 