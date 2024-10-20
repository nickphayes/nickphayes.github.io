---
layout: post
title: "working title"
categories: "Machine Learning"
featImg: 
excerpt: "Mathematical Intuitions for Modern ML"
permalink: "working-link"
style: 
---

---
### the goal
trace chronology of modern ML models via mathematical intuitions

Training a modern ML system[^a] typically involves four main components: a dataset, a model, a loss function, and an optimizer.

Parameters are what a machine actually learns. 
Model architectures control how parameters interact with one another. 
Optimizers control how we update the parameters during each training step. 
Loss functions define how we evaluate model performance on the objective, 
doing most of the heavy lifting for guiding model behavior towards a particular task. 
Lastly, our target behavior (i.e. objective) and data inform the design of each of the above.  

---
# Main Title

Some quick notation:
- $$ x $$ | inputs
- $$ y $$ | labels
- $$ \theta $$ | parameters
- $$ m(\theta) $$ | a model defined on some set of parameters
- $$ m(\theta; x) $$ | a model define on some set of parameters and evaluated on inputs $$ x $$
- $$ \mathcal{L} $$ | a loss function

Things we ignore here: input embeddings, layer normalization, output layers

## Basics

Training a model is equivalent to finding the set of parameters that minimizes the loss function:

$$
\theta^* := \underset{\theta \in \Theta}{\operatorname{argmin}}  \mathcal{L} (m(\theta; x), y)
$$

subject to some iterative update rule defined by the optimizer. 

$$
\theta^{k+1} = \theta^k + \alpha^k s^k 
$$

## Linear Regression
**Learnable Parameters**: $$ W, b $$

For $$ x \in \mathbb{R}^n $$, $$ b \in \mathbb{R}^m $$, and $$ W \in \mathbb{R}^{m \times n}$$, linear models take the following form:

$$
\begin{align*}
\hat{y} = Wx + b
\end{align*}
$$

$$ \theta $$ consists of $$\{ W, b \} $$, which define linear transformations of the input data. It is sufficient to consider only the shallow case, 
as any composition of linear layers can be described as a single linear transformation. We can think of $$ W $$ as scaling the input space, while $$ b $$ translates the data in this newly defined space. Together, $$ W $$ and $$ b $$ define a hyperplane. For classification tasks, this represents a decision boundary (suitable for linearly separable data), while for regression tasks, this represents a continouous output surface corresponding to model predictions. 

## Logistic Regression
**Learnable Parameters**: $$ W, b $$

Logistic models start with the following form:

$$
\begin{align*}
\sigma(z) &= \sigma(Wx + b)
\end{align*}
$$

where $$ \sigma $$ represents either a sigmoid (or softmax) activation for binary (or multi-class) classification tasks[^b]. The sigmoid function is defined as 

$$
\sigma(z)= \frac{1}{1 + e^{-z}}
$$

which has the effect of squashing the output into the interval $$(0, 1)$$. Using some threshold $$ \tau \in (0,1) $$, models thus make predictions by evaluating $$ \sigma(z) \geq \tau$$. 

$$
\hat{y} = \begin{cases} 
    1 & \text{if } \sigma(z) \geq \tau \\
    0 & \text{if } \sigma(z) < \tau
\end{cases}
$$

For the multi-class case, we instead use an element-wise softmax activation, which in some sense can be seen as a generalization of the sigmoid function. In this case, 

$$ 
\sigma(z_k)= \frac{e^{z_k}}{\sum_{j=1}^{m} e^{z_j}} 
$$

which ensures both $$ \sigma(z_k) \in (0, 1)$$ and $$ \sum_{j=1}^{m} \sigma(z_j) = 1$$. It is helpful (but misleading) to think of the resulting vector entries as probabilities. Predictions are made by determining the class (index) with the highest "probability".

$$
\hat{y} = \underset{k}{\operatorname{argmax}} \sigma(z)
$$

As most machine learning objectives are abstracted into multi-class classification problems, this is an extremely common output layer. 

## Deep Neural Network
**Learnable Parameters**: $$ W_k, b_k $$ for $$ k = 0, 1, \dots , l$$ 

A vanilla deep neural network can be seen as a stack of linear layers with nonlinear activations applied (usually element-wise) in-between layers.
With $$ z_0 = W_0x + b_0 $$, a network with $$ l $$ layers is composed such that

$$
\begin{align*}
z_1 &= W_1 h_{0} + b_1 \\
h_1 &= \sigma (z_1) \\
&\vdots \\
z_l &= W_l h_{l-1} + b_l \\
h_l &= \sigma (z_l)
\end{align*}
$$

Choice of the activation function $$ \sigma $$ is motivated by empirical rather than theoretical reasons.
Historically popular choices include tanh, sigmoid, and ReLU, though current SOTA models seem to prefer non-monotonic, "leaky"
activation functions (e.g. GELU, LeakyReLU, SiLU). 

## Convolutional Neural Network

## Transformer

An encoder block begins with some input $$ x' $$, representing an embedded version of $$ x $$.




[^a]: This is a fairly narrow definition for what constitutes machine learning, lending itself more easily to (semi-)supervised machine learning problems. This is not the only way in which machines can "learn"; it is, however, the dominant paradigm that can be used to trace major milestones in ML. 
[^b]: Logistic models are typically reserved for classification tasks.
[^c]: Embeddings are notably ignored here, as we assume the input to the model is not $$ x $$, rather some embedded representation $$ x' $$ 