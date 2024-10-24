---
layout: post
title: "From Linear Regression to Transformers"
categories: "Machine Learning"
featImg: neuralnetwork.png
excerpt: "The math behind modern ML architectures"
permalink: "math-of-modern-ML"
style: 
---

---
### Introduction
Training a modern deep learning model[^a] typically involves four main components: a model architecture, a loss function, an optimizer, and a dataset.
A few quick points:
1. Parameters are what a machine is learning in "machine learning". 
2. Model architectures control how parameters interact with one another. 
3. Loss functions define how we evaluate model performance on our objective.
4. Optimizers control how we update the parameters in response to loss function feedback. 
5. Our target behavior (i.e. objective) and data inform the design of each of the above.  

This post is focused on **model architectures**---specifically, developing mathematical intuitions for modern deep learning. 
Below, I 
mathematically formulate increasingly complex models, starting with linear regression and building to today's transformer-based
paradigm. 

**Note**: *this post is not exhaustive.* I have identified what I understand to be the main paradigms contributing to current,
transformer-based approaches. Notable exclusions from the zoo of ML models 
include SVMs, RNNs, LSTMs[^f], RL methods, Bayesian models, GANs, VAEs, and KANs.

For any questions regarding notation, please see the [notation](#notation) section at the bottom. 



---

# Basics

Training a model is equivalent to finding the set of parameters that minimizes a loss function:

$$
\theta^* := \underset{\theta \in \Theta}{\operatorname{argmin}}  \mathcal{L} (m(\theta; X), Y)
$$

subject to some iterative update rule defined by the optimizer. 

$$
\theta^{k+1} = \theta^k + \alpha^k s^k 
$$

## Model
For the purposes of this post, I will use "model" to refer to core architecture *excluding output layers and (for the most part) embedding layers*.
The model thus defines how a (possibly embedded) input is propagated forwards to produce the final set of model logits. We can then define the 
output layers as the mapping from logits to overall model prediction. 
## Output Layers
For regression tasks, the output layer often takes the form of a simple linear mapping from the logit space to the target prediction space. For
classification tasks, we might also require a sigmoid or softmax activation function. The sigmoid function is defined as 

$$
\sigma(z)= \frac{1}{1 + e^{-z}}
$$

where we use $$ z = y_L $$ for ease of notation. This has the effect of squashing a single-valued output into the interval $$(0, 1)$$. Using some threshold $$ \tau \in (0,1) $$, models thus make predictions by evaluating $$ \sigma(z) \geq \tau$$. 

$$
\hat{y} = \begin{cases} 
    1 & \text{if } \sigma(z) \geq \tau \\
    0 & \text{otherwise }
\end{cases}
$$

For the multi-class case, we instead use an element-wise softmax activation, which in some sense can be seen as a generalization of the sigmoid function. In this case, for $$ z \in \mathbb{R}^m $$

$$ 
\sigma(z_k)= \frac{e^{z_k}}{\sum_{j=1}^{m} e^{z_j}} 
$$

which ensures both $$ \sigma(z_k) \in (0, 1)$$ and $$ \sum_{j=1}^{m} \sigma(z_j) = 1$$. It is helpful (but misleading) to think of the resulting vector entries as probabilities. Predictions are made by determining the class (index) with the highest "probability".

$$
\hat{y} = \underset{k}{\operatorname{argmax}} \sigma(z)
$$

As most machine learning objectives are abstracted into multi-class classification problems, this is an extremely common output layer. 

# Model Architectures
## Linear Regression
**Learnable Parameters**: $$ \theta = \{W, b \}$$

For $$ x \in \mathbb{R}^n $$, $$ b \in \mathbb{R}^m $$, and $$ W \in \mathbb{R}^{m \times n}$$, linear models take the following form:

$$
\begin{align*}
y_L = Wx + b
\end{align*}
$$

It is sufficient to consider only the shallow case, 
as any composition of linear layers can be described as a single linear transformation. We can think of $$ W $$ as scaling the input space, while $$ b $$ translates the data in this newly defined space. Together, $$ W $$ and $$ b $$ define a hyperplane. For classification tasks, this represents a decision boundary (suitable for linearly separable data), while for regression tasks, this represents a continouous output surface corresponding to model predictions. 

## Logistic Regression
**Learnable Parameters**: $$ \theta = \{W, b \}$$

For $$ x \in \mathbb{R}^n $$, $$ b \in \mathbb{R}^m $$, and $$ W \in \mathbb{R}^{m \times n}$$, logistic models take the following form:

$$
\begin{align*}
y_L &= \sigma(Wx + b)
\end{align*}
$$

where $$ \sigma $$ represents either a sigmoid (or softmax) activation for binary (or multi-class) classification tasks[^b].

## Vanilla Deep Neural Network
**Learnable Parameters**: $$ \theta = \{ W_k, b_k $$ for each layer $$\}$$ 

A vanilla deep neural network (also called an MLP --- multilayer perceptron) 
can be seen as a stack of linear layers with nonlinear activations applied (usually element-wise) in-between layers.
With $$ z_0 = W_0x + b_0 $$, a network with $$ l+1 $$ layers is composed such that

$$
\begin{align*}
h_0 &= \sigma(z_0) \\
z_1 &= W_1 h_{0} + b_1 \\
h_1 &= \sigma (z_1) \\
&\vdots \\
z_l &= W_l h_{l-1} + b_l \\
h_l &= \sigma (z_l) \\
\downarrow \\
y_L &= h_l
\end{align*}
$$

Choice of the activation function $$ \sigma $$ is motivated mostly by empirical rather than theoretical reasons.
Historically popular choices include tanh, sigmoid, and ReLU, though current state-of-the-art models seem to prefer non-monotonic, "leaky"
activation functions (e.g. GELU, LeakyReLU, SiLU). For simplicity, we also assume the vanilla MLP is fully connected, though in practice this is not
always the case (e.g. dropout layers). For further ease of notation, we'll use a shorthand notation to denote the model described above:

$$
y_L = MLP(x)
$$

## Convolutional Neural Network
**Learnable Parameters**: $$ \theta = \{ W_k, b_k \text{ for each MLP layer; }f_l, b_l, \text{ for each filter}\}$$

Convolutional Neural Networks (CNNs) were originally inspired by attempts to capture localized pixel relationships during computer vision tasks. 
Image pixels don't exist in a vacuum; rather, they contribute meaning to the image as a function of their context (i.e., other surrounding pixels). 
Rather than manually defining how to perform localized feature extraction, CNNs (specifically the convolution part) learn these for us. 
Otherwise, downstream portions of CNNs beyond convolution are essentially the same as vanilla MLPs. We can thus think of CNNs as two distinct 
components joined together: a set of feature extractors, and an MLP. 

The feature extractor starts by learning a set of feature maps. To generate a feature map, we slide a matrix over subsections of the input
and perform element-wise matrix multiplication. We call one of these sliding matrices a filter or kernel, while where/how we move the filter around
the input is controlled by its padding and stride-length. Each time we convolve a filter with the input, we obtain a new matrix, whose elements 
are summed and used to populate the feature map. Feature maps are therefore matrices whose entries correspond to the summed entries of matrices obtained
from convolving the filter with different subsections of the input[^e]. For simplicity, we assume a stride-length of one and zero padding. 

$$
\begin{align*}
\phi_{i, j} &= \sum_{m=0}^{k_h}\sum_{n=0}^{k_w} x_{i+m, j+n} f_{m, n} 
\end{align*}
$$

Once we have a feature map, we add a bias and nonlinear activation function before feeding into the aggregation component.

$$
\begin{align*}
\Phi = \sigma(\phi + b)
\end{align*}
$$

Aggregation involves reducing the dimensionality of the learned feature map, typically through a pooling layer (e.g. max-pooling, average-pooling). 
For example, during max-pooling, we populate yet another matrix by sliding a window over $$ \Phi $$ and 
extracting the maximum element from each window.

$$
\begin{align*}
F = Pool(\Phi)
\end{align*}
$$

Here $$ F $$ denotes the final result from convolving and aggregating one filter over the input. In practice, we can do this for 
$$ n $$ filters, generating $$ n $$ feature extractors. For $$ l = 1, \dots, n$$, 

$$
\begin{align*}
F^{(l)} &= Pool(\Phi^{(l)}) \\
&= Pool(\sigma(\phi^{(l)} + b^{(l)})) \\
&\text{ }\\
&\text{where} \\
&\text{ }\\
\phi_{i, j}^{(l)} &= \sum_{m=0}^{k_h}\sum_{n=0}^{k_w} x_{i+m, j+n} f_{m, n}^{(l)}
\end{align*}
$$

At this stage, we have a set of $$ n $$ feature extractors which are to be fed into an MLP. There is some nuance as to how
we choose to flatten and combine inputs, but a basic approach would be to flatten each $$ F^{(l)} $$ and concatenate the results, generating a large input vector
for an MLP. 

$$
\begin{align*}
F' &= Concat (\{Flatten(F^{(1)}), \dots,Flatten(F^{(n)}) \}) \\
& \downarrow \\
y_L &= MLP(F')
\end{align*}
$$

Written more concisely, a generic CNN takes the following form:

$$
\begin{align*}
y_L = MLP(Conv(x))
\end{align*}
$$

where we use $$ Conv(x) $$ to denote the process of concatenating and flattening learned feature maps. 

## Transformer

**Learnable Parameters (per transformer block)**
<!-- $$ \theta = \{ W_{E}, W_{U}, \text{ for embedding/unembedding; }W_{QK}, W_{V}, \text{ for each attention head; }W_k, b_k \text{ for each MLP layer}\}$$ -->

$$ 
\begin{align*}
\theta = \{ W_{E}, W_{U}, \text{ for embedding/unembedding;} \\ 
W_{QK}, W_{V}, \text{ for each attention head;} \\ 
W_k, b_k \text{ for each MLP layer} \}
\end{align*}
$$


The backbone of the transformer architecture is the **residual stream**, which carries information through the model. 
It is initialized with an embedded representation of our input, then subsequently modified by successive blocks. 
For each block (ignoring normalization layers), we add the outputs from multi-head attention and an MLP to the residual stream. 
We then repeat successive blocks before applying an unembedding[^c] layer to obtain output logits. 
We'll break down each component below. 

The **embedding layer** is a simple weight matrix applied to the input. Conceptually, we can think of $$ W_E $$ as
defining some lookup table that maps $$ x \rightarrow x' $$.  

$$
\begin{align*}
  x' = W_Ex
\end{align*}
$$

The original formulation for **single-head attention** computes keys, queries, and values from three 
separate weight matrices. Attention scores are the normalized matrix product of queries and keys, 
which are then fed into a softmax activation and multiplied with the values. We can think of the attention scores
as learning which information to "attend" to, while the value matrix
determines how much of that information to actually "copy" to the residual stream. 
Using $$ d_k $$ to represent the missing dimension of the attention matrices, we have

$$
\begin{align*}
  Q &= W_Qx' \\
  K &= W_Kx' \\
  V &= W_Vx' \\
  &\downarrow \\
  a &=\text{softmax}(\frac{Q^TK}{\sqrt{d_k}}) V^T
\end{align*}
$$

This style of attention is referred to as scaled dot-product attention, and though popular, is not the only method for calculating 
attention scores (other examples include additive attention and unscaled dot-product attention). Further inspection shows us we can express this directly in terms of just two learnable weight matrices[^d]:
a key-query matrix and a value matrix. 

$$
\begin{align*}
    a &=\text{softmax}(\frac{Q^TK}{\sqrt{d_k}}) V^T \\
    &= \text{softmax}(\frac{x'^TW_{Q}^TW_Kx'}{\sqrt{d_k}}) x'W_V \\
    &= \text{softmax}(\frac{x'^TW_{QK}x'}{\sqrt{d_k}}) x'W_V
\end{align*}
$$

**Multi-head attention** is exactly what it sounds like: we repeat the same process for multiple attention heads. 
To combine results and return to the dimension of the residual stream, we concatenate all attention heads 
together and scale by yet another matrix. For $$ n $$-head attention, using $$ i $$ to index heads, we have

$$
\begin{align*}
  a_i &= \text{softmax}(\frac{x'^T(W_{QK})^{(i)}x'}{\sqrt{d_k}}) x'W_V^{(i)} \\
  a_{MH} &= \text{Concat}(a_i, \dots, a_n)W_{MH}
\end{align*}
$$

Depending on the pre-training objective (e.g. masked language modeling, next-token prediction, knowledge representation etc.) 
and model design choices, there are additional nuances regarding multi-head attention. For instance, encoder and decoder models differ in 
how the softmax activation is applied (with/without autoregressive masking); some may choose to directly add the output of each 
attention head to the residual stream rather than concatenating and re-weighting; some schema may even apply an additional re-weighting to each 
head pre-concatenation. These nuances aside, the core flow remains the same. 

The **MLP layer** is exactly as described [above](#deep-neural-network): a stack of linear layers separated by nonlinear activations. 
We'll henceforth denote the output of the MLP as $$ x_{MLP} $$. Putting it all together, each block simply adds the multi-head attention and 
MLP output to the residual stream. 

$$
\begin{align*}
r^{(0)} &= W_Ex \\
r^{(1)} &= r_0 + a_{MH}^{(1)} + x_{MLP}^{(1)} \\
\end{align*}
$$

For $$n$$-inner blocks feeding into an **unembedding layer** (essentially a reverse lookup table), the full flow of a completed
transformer block takes the following form:

$$
\begin{align*}
x \\
\downarrow \\
r^{(0)} &= W_Ex \\
r^{(1)} &= r_0 + a_{MH}^{(1)} + x_{MLP}^{(1)} \\
\vdots \\
r^{(n)} &= r^{(n-1)} + a_{MH}^{(n)} + x_{MLP}^{(n)} \\
\downarrow \\ 
T(x) &= W_Ur^{(n)}
\end{align*}
$$

where $$ T(x) $$ denotes the output from a single, generic transformer block. Transformer based models typically stack many of these blocks in sequence, while reserving the majority ($$ \approx \frac{5}{6} $$) of learnable parameters for MLPs, with the remaining parameters ($$\approx \frac{1}{6}$$) reserved for attention. Barring some design choices bespoke to the target objective, as well as some additional performance considerations (e.g. normalization layers, parallelization), that's all there is to it.

# Concluding Thoughts

You may have heard fundamental models described as a big stack of matrix multiplication and linear algebra. Now, you have also *seen* the core architectural block of fundamental models---the transformer---described as a big stack of matrix multiplcation and linear algebra. Beyond this big stack of matrices, here's what's actually exciting about reducing models down to this level of mathematical granularity: we can build a better understanding of *how the hell models actually work*. For instance, we can 
- use insights from knot theory and differentiable manifolds to conceptualize models as untangling knots. Read more [here.](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- use insights from sparse coding and information compression to recover seemingly interpretable features from compressed representations. Read more [here.](https://transformer-circuits.pub/2023/monosemantic-features)
- use insights from numerical optimization and numerical stability to visualize model internals. Read more [here.](https://distill.pub/2021/multimodal-neurons/)

These are just a few of my favorites, but there's no shortage of ways in which mathematics can be used to enhance our ability to design, train, and understand sophisticated models. And it all starts with a big stack of linear algebra. 

---

## Notation

|                  |                                               |
|:-----------------|:----------------------------------------------------|
| $$ x $$          | input                                             |
| $$ X $$          | set of inputs                                           |
| $$ x' $$          | embedded input                                             |
| $$ y $$          | ground truth labels                                              |
| $$ Y $$          | set of ground truth labels                                              |
| $$ y_L $$          | logits (pre-output layer)                                               |
| $$ \hat{y} $$          | predictions (post-output layer)                                               |
| $$ \theta $$          | learnable parameters                                               |
| $$ \Theta $$          | feasible set from which $$ \theta $$ is drawn                                               |
| $$ \alpha $$          | step direction                                               |
| $$ s $$          | step size (i.e. learning rate)                                               |
| $$ \sigma $$          | nonlinear activation function                                               |
| $$ W $$          | weight matrix                                               |
| $$ b $$          | bias term                                              |
| $$ f $$          | kernel matrix (filter)                                              |
| $$ k_h $$          | kernel height                                              |
| $$ k_w $$          | kernel width                                              |
| $$ \phi $$          | feature map                                             |
| $$ \Phi $$          | post-activation feature map                                             |
| $$ F $$          | aggregated feature map                                             |
| $$ F' $$          | output post convolution, pre-MLP in a CNN                                       |
| $$ r $$          | residual stream                                      |
| $$ W_Q $$          | query weight matrix                                               |
| $$ W_K $$          | key weight matrix                                               |
| $$ W_V $$          | value weight matrix                                               |
| $$ W_{QK} $$          | query-key weight matrix                                               |
| $$ W_E $$          | embedding weight matrix                                               |
| $$ W_U $$          | unembedding weight matrix                                               |
| $$ m(\theta; X) $$  | a model defined on some set of parameters with inputs x        |
| $$ \mathcal{L} $$| loss function                                     |

---

[^a]: This implies a fairly narrow definition for what constitutes machine learning, lending itself more easily to (semi-)supervised machine learning problems. This is not the only way in which machines can "learn"; it is, however, one of the dominant paradigms that can be used to trace major milestones in ML.
[^b]: Logistic models are typically reserved for classification tasks.
[^c]: Note here that I use the language "embedding/unembedding" rather than "encoding/decoding." This is to avoid misleading jargon, as encoder-only, decoder-only, and encoder-decoder models all do this embedding/unembedding. The encoder/decoder typology is rather split based on whether/how masking is applied during multihead attention. Decoders use autoregressive masking, while encoders do not. This makes the former particular well-suited for generative tasks, while the latter is likely better for learning latent representations of the data. 
[^d]: While this might make the math easier to digest, it could also change the way we count parameters. Given that most AI safety frameworks and responsible scaling policies use parameter and flop counts as a proxy for model capability, this quickly becomes relevant to technical AI governance. 
[^e]: Although accurate, this is an unfortunately complex way to describe how a feature map is generated in just one sentence. The added depth and composition of operations makes it increasingly difficult to succintly describe components of the model in terms of the original input. 
[^f]: In particular, RNNs and LSTMs arguably deserve to be in this post, as they were critical to the development of modeling sequential data with neural networks. They are intentionally ommitted, as I believe a separate post dedicated purely to advancements in modeling sequential data is prudent (i.e. time-series data, language data, etc.). 

