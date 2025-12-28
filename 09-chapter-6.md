Chapter 6: Relativity and Time: Recurrent Networks and TransformersIn This ChapterThe limitation of "Snapshot" IntelligenceRecurrent Neural Networks (RNNs): The Flywheel of AIThe Vanishing Gradient Problem (Time Decay)The Transformer Revolution: Bending Time with AttentionLab: Coding the Mechanism Behind ChatGPTIn the previous chapters, our AI was static. We showed it a picture of a cat, and it said "Cat." It didn't matter if we showed the picture yesterday or today. The network had no memory of the past.But language, music, and the stock market are defined by Time. You cannot understand the word "bank" unless you know if the previous word was "river" or "piggy."To handle this, we need architectures that possess Inertia. We need systems that remember the state of the universe from $t-1$ to calculate the state at $t$.The Arrow of Time: Recurrent Neural Networks (RNNs)A standard Feed-Forward network (like our MLP or CNN) is a one-way street. Input $\rightarrow$ Output.A Recurrent Neural Network (RNN) adds a loop. It feeds its own output back into itself as input for the next step.The Physics of the Hidden State ($h$)Think of the hidden state as the momentum of the conversation.Input ($x_t$): The new word you just heard.Previous Hidden State ($h_{t-1}$): The context or "vibe" of the sentence so far.New Hidden State ($h_t$): The updated context.$$h_t = \tanh(W \cdot [h_{t-1}, x_t] + b)$$This loop allows the network to "read." It processes the first word, updates its mental state, then processes the second word carrying that state forward.WARNING: The Exploding/Vanishing Gradient Problem.When you unroll an RNN over a long sentence (say, 100 words), you are effectively creating a 100-layer deep network.As we learned in Chapter 3, gradients decay over distance. In an RNN, gradients decay over Time. The network forgets the beginning of the sentence by the time it reaches the end. It has short-term memory loss.The Valve System: LSTMsTo fix the memory loss, physicists (and computer scientists) invented the Long Short-Term Memory (LSTM) unit.If a standard RNN is a leaky bucket, an LSTM is a pressurized tank with valves.Forget Gate: Should I dump the old information? (Entropy flush).Input Gate: Is this new information worth saving?Output Gate: How much of this internal state should I reveal to the world right now?These "gates" are learnable. The network learns when to forget and when to remember.The Revolution: Transformers and RelativityRNNs and LSTMs have a fatal flaw: They are sequential. To understand word #100, you must process words #1 through #99 first. This is slow (high latency) and computationally inefficient.In 2017, the paper "Attention Is All You Need" changed the world. It introduced the Transformer.The Physics of AttentionImagine a sentence not as a line of dominoes, but as a solar system. Every word is a planet.In a Transformer, every word looks at every other word simultaneously, regardless of how far apart they are.RNN: "The... (wait)... cat... (wait)... sat." (Sequential/Linear).Transformer: The words "Cat" and "Sat" exert a gravitational pull on each other instantly. The distance in the sentence doesn't matter. The network "bends time" to connect related concepts.The Mechanism: Query, Key, and ValueHow does a word "attend" to another? It uses a database retrieval mechanism analogous to a filing system.Query ($Q$): What am I looking for? (e.g., The word "Bank" asks: "Are there any adjectives nearby describing water?")Key ($K$): What describes me? (e.g., The word "River" shouts: "I am related to water!")Value ($V$): The actual content. (The meaning vector of "River").The Attention Formula:$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$This formula calculates the Resonance between words. If the Query matches the Key, the "energy" (score) is high, and the network pays attention to that Value.Lab 6.1: The Self-Attention EngineLet's build the core engine of ChatGPT—the Self-Attention mechanism—from scratch in PyTorch. We will see how words calculate affinity for each other.The Experiment:Pythonimport torch
import torch.nn.functional as F
import math

# 1. The Inputs (Embeddings)
# Imagine 3 words: "The", "Cat", "Sat"
# We represent each word as a vector of size 4 (d_model=4)
# Shape: [Batch=1, Sequence_Length=3, Embedding_Dim=4]
inputs = torch.tensor([
    [[1.0, 0.0, 1.0, 0.0],  # "The"
     [0.0, 1.0, 0.0, 1.0],  # "Cat"
     [1.0, 1.0, 1.0, 1.0]]  # "Sat"
])

# 2. Define Linear Layers for Q, K, V
# In a real model, these are learned weights. Here we fix them for clarity.
d_model = 4
query_layer = torch.nn.Linear(d_model, d_model, bias=False)
key_layer   = torch.nn.Linear(d_model, d_model, bias=False)
value_layer = torch.nn.Linear(d_model, d_model, bias=False)

# Generate Q, K, V matrices
Q = query_layer(inputs)
K = key_layer(inputs)
V = value_layer(inputs)

# 3. Calculate Energy Scores (Dot Product)
# We want to multiply Q by K_transpose to find similarities.
# If Q and K align, the dot product is high (High Attention).
scores = torch.matmul(Q, K.transpose(-2, -1))

# Normalize by square root of dimension (Temperature Scaling)
# This prevents the gradients from exploding in high dimensions.
scores = scores / math.sqrt(d_model)

print("Raw Affinity Scores (Pre-Softmax):")
print(scores)

# 4. Apply Softmax (Probability Distribution)
# This converts raw scores into percentages (0.0 to 1.0).
attention_weights = F.softmax(scores, dim=-1)

print("\nAttention Map (Who is looking at whom?):")
# Rows = Current Word, Cols = Word being looked at
print(attention_weights)

# 5. Aggregate Values
# Multiply weights by Values to get the final context vector.
context = torch.matmul(attention_weights, V)

print("\nFinal Context Vectors:")
print(context)
Analyzing the Results:The attention_weights matrix is the most important part.If Row 1, Col 2 is high ($0.9$), it means Word 1 ("The") is paying 90% of its attention to Word 2 ("Cat"). It has realized that "The" is modifying "Cat."This mechanism allows the model to dynamically route information between words.THEORY LAB: Positional Encoding.Notice that in the math above, there is no reference to the order of words ($1, 2, 3$). If we shuffled the words, the dot products ($Q \cdot K$) would be identical.Since Transformers process everything simultaneously, they are technically invariant to permutation.To fix this, we inject a "Positional Encoding" signal (sine waves of different frequencies) into the input vectors so the model knows that "Cat" came before "Sat."The Rise of LLMsBy stacking these Attention blocks 96 layers high and training them on the entire internet, we get GPT-4. But fundamentally, it is just this simple mechanism: calculating the gravitational pull between words to predict the next token.This concludes Part II. We have built the brain (MLP), the eyes (CNN), and the language center (Transformer).Now, we enter Part III, where physics gets weird. We are going to study Chaos, Entropy, and Creation.