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
- $$ x $$ \| inputs
- $$ y $$ \| labels
- $$ \theta $$ \| parameters
- $$ m(\theta) $$ \| a model defined on some set of parameters
- $$ m(\theta; x) $$ \| a model define on some set of parameters and evaluated on inputs $$ x $$
- $$ \mathcal{L} $$ \| a loss function

## Basics

Training a model is equivalent to finding the set of parameters that minimizes a loss function:

$$
\theta^* := \underset{\theta \in \Theta}{\operatorname{argmin}}  \mathcal{L} (m(\theta; x), y)
$$

subject to some iterative update rule defined by the optimizer. 

$$
\theta^{k+1} = \theta^k + \alpha^k s^k 
$$
### Model
For our purposes here, I will use "model" to refer to core architecture *excluding output layers and (for the most part) embedding layers*.
The model thus defines how a (possibly embedded) input is propagated forwards to produce the final set of model logits. We can then define the 
output layers as the mapping from logits to overall model prediction. 
### Output Layers
For regression tasks, the output layer often takes the form of a simply linear mapping from the logit space to the target prediction space. For
classification tasks, we might also require a sigmoid or softmax activation function. The sigmoid function is defined as 

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

A vanilla deep neural network (also called an MLP -- multilayer perceptron) 
can be seen as a stack of linear layers with nonlinear activations applied (usually element-wise) in-between layers.
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

Choice of the activation function $$ \sigma $$ is motivated mostly by empirical rather than theoretical reasons.
Historically popular choices include tanh, sigmoid, and ReLU, though current SOTA models seem to prefer non-monotonic, "leaky"
activation functions (e.g. GELU, LeakyReLU, SiLU). We also assume the vanilla MLP is fully connected, though in practice this is not
always the case (e.g. dropout layers). 

## Convolutional Neural Network
**Learnable Parameters**: $$ \theta = \{ W_k, b_k \text{ for each MLP layer; }f_l, b_l, \text{ for each filter}\}$$

Convolutional Neural Networks (CNNs) were originally inspired by attempts to capture localized pixel relationships during computer vision tasks. 
Image pixels don't exist in a vacuum; rather, they contribute meaning to the image as a function of their context (i.e., other surrounding pixels). 
Rather than manually defining how to perform localized feature extraction, CNNs (specifically the convolution part) learn these for us. 
Otherwise, downstream portions of CNNs beyond convolution are essentially the same as vanilla MLPs. We can thus think of CNNs as two distinct 
components joined together: a set of feature extractors, and an MLP. 

The feature extractor starts by learning a set of *feature maps*. To generate a feature map, we slide a matrix over subsections of the input
and perform element-wise matrix multiplication. We call one of these sliding matrices a *filter* or *kernel*, while how we move the filter around
the input is controlled by its *padding* and *stride-length*. Each time we convolve a filter with the input, we obtain a new matrix, whose elements 
are summed and used to populate the *feature map*. Feature maps are thus matrices whose entries correspond to the summed entries of matrices obtained
from convolving the filter over different locations of the input. As you can see, the added depth and composition of operations makes it increasingly 
difficult to describe components of the model in terms of the original input. 

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
&\text{where} \\
\phi_{i, j}^{(l)} &= \sum_{m=0}^{k_h}\sum_{n=0}^{k_w} x_{i+m, j+n} f_{m, n}^{(l)}
\end{align*}
$$

At this stage, we have a set of $$ n $$ feature extractors $$ F^{(l)} $$, which are to be fed into an MLP. There is some nuance as to how
we choose to flatten and combine inputs, but a basic approach would be to flatten each $$ F^{(l)} $$ and concatenate the results, generating a large input vector
for an MLP. 

$$
\begin{align*}
F' &= Concat (Flatten(F^{(l)})) \\
\text{for } l &= 1, \dots, n \\
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

where we use $$ Conv(x) $$ to represent the process of concatenating and flattening learned feature maps. 

## Transformer

**Learnable Parameters (per transformer block)**
$$ \theta = \{ W_k, b_k \text{ for each MLP layer; }W_{QK}, W_{V}, \text{ for each attention head}\}$$


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
  a &=\text{softmax}(\frac{Q^TK}{d_k}) V^T
\end{align*}
$$

Further inspection shows us we can express this directly in terms of just two learnable weight matrices[^d]:
a key-query matrix and a value matrix. 

$$
\begin{align*}
    a &=\text{softmax}(\frac{Q^TK}{d_k}) V^T \\
    &= \text{softmax}(\frac{x'^TW_{Q}^TW_Kx'}{d_k}) x'W_V \\
    &= \text{softmax}(\frac{x'^TW_{QK}x'}{d_k}) x'W_V
\end{align*}
$$

**Multi-head attention** is exactly what it sounds like: we repeat the same process for multiple attention heads. 
To combine results and return to the dimension of the residual stream, we concatenate all attention heads 
together and scale by yet another matrix. For $$ n $$-head attention, using $$ i $$ to index heads, we have

$$
\begin{align*}
  a_i &= \text{softmax}(\frac{x'^T(W_{QK})^{(i)}x'}{d_k}) x'W_V^{(i)} \\
  a_{MH} &= \text{Concat}(a_i, \dots, a_n)W_{MH}
\end{align*}
$$

Depending on the pre-training objective (e.g. masked language modeling, next-token prediction, knowledge representation etc.) 
and model design choices, there are additional nuances regarding multi-head attention. For instance, encoder and decoder models differ in 
how the softmax activation is applied (with/without autoregressive masking), while some may also choose to directly add the output of each 
attention head to the residual stream rather than concatenating and re-weighting. These nuances aside, the core flow remains the same. 

The **MLP layer** is exactly as described [above](#deep-neural-network): a stack of linear layers separated by nonlinear activations. 
We'll henceforth denote the output of the MLP as $$ x_{MLP} $$. Putting it all together, each block simply adds the multi-head attention and 
MLP output to the residual stream. 

$$
\begin{align*}
r^{(0)} &= W_Ex \\
r^{(1)} &= r_0 + a_{MH}^{(1)} + x_{MLP}^{(1)} \\
\end{align*}
$$

For $$n$$-blocks feeding into an **unembedding layer** (essentially a reverse lookup table), the full flow takes the following form:

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



DIMENSION CHECK




[^a]: This is a fairly narrow definition for what constitutes machine learning, lending itself more easily to (semi-)supervised machine learning problems. This is not the only way in which machines can "learn"; it is, however, the dominant paradigm that can be used to trace major milestones in ML. 
[^b]: Logistic models are typically reserved for classification tasks.
[^c]: Note here that I use the language "embedding/unembedding" rather than "encoding/decoding." This is to avoid misleading jargon, as encoder-only, decoder-only, and encoder-decoder models all do this embedding/unembedding; the models are rather split based on whether/how masking is applied during multihead attention. Decoders use autoregressive masking, while encoders do not. This makes the former particular well-suited for generative tasks, while the latter is likely better for learning latent representations of the data. 
[^d]: While this might make the math easier to digest, it could also change the way we count learnable parameters.