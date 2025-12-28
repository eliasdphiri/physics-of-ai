Chapter 8: The Boltzmann Distribution of Words: Large Language Models (LLMs)In This ChapterWords as Atoms (Tokens)The "Next Word" as a Quantum Wave FunctionTemperature and Sampling: Controlling the State of MatterLab: Building a Maxwell-Boltzmann Sampler for TextWhen you chat with an AI like ChatGPT or Claude, it feels like you are talking to a conscious being. It argues, jokes, and apologizes.But physically, an LLM is simply a Statistical Engine. It is a machine that predicts the next "atom" in a sequence based on the thermodynamic state of the previous atoms. It doesn't "know" facts; it calculates the probability of a word appearing next to another word, based on the statistical average of the entire internet.The Atom of Language: The TokenComputers don’t read English. They read numbers. Before an LLM sees your text, the text is broken down into Tokens.A token is the fundamental unit of meaning. It can be a whole word ("apple"), a part of a word ("ing"), or even a space.Mass: ~4 characters per token.Vocabulary: The Periodic Table of the LLM. GPT-4 has a vocabulary of roughly 100,000 distinct tokens.The Probability Wave FunctionWhen you type "The cat sat on the...", the LLM does not immediately pick "mat."Instead, it calculates a Probability Distribution over its entire vocabulary. It assigns an energy score (logit) to every possible next word."mat": 15% probability"couch": 10% probability"floor": 8% probability"moon": 0.00001% probability (Unlikely, but non-zero)At this exact moment, the "next word" exists in a superposition of all possible words. It hasn't collapsed into a single reality yet. It is a wave function.Collapsing the Wave: The Sampling StrategyTo generate text, we must observe the system. We must pick one word. How we pick that word determines the "creativity" or "intelligence" of the model.This is where Thermodynamics comes in.Greedy Decoding (Absolute Zero, 0 Kelvin)We always pick the word with the highest probability.Physics: The system is frozen in a crystal lattice.Result: The model is robotic, repetitive, and boring. It gets stuck in loops ("I don't know, I don't know, I don't know").Temperature Sampling ($T > 0$)We introduce heat. We use the Boltzmann Distribution (Softmax with Temperature) to redistribute the probabilities.$$P_i = \frac{\exp(E_i / T)}{\sum \exp(E_j / T)}$$$P_i$: Probability of word $i$.$E_i$: Energy (Logit) of word $i$.$T$: Temperature.Low Temperature ($T < 1$): We sharpen the peaks. The likely words become more likely. The unlikely words disappear. (Solid/Liquid state). Good for coding and math.High Temperature ($T > 1$): We flatten the curve. The difference between "mat" and "moon" shrinks. The model becomes chaotic. (Gas/Plasma state). Good for poetry and brainstorming.WARNING: Hallucinations.If the temperature is too high, the system enters a plasma state where bonds between logical concepts break. The model will confidently state that "The cat sat on the... algorithm." This is a hallucination caused by high entropy.Lab 8.1: Maxwell’s DictionaryWe will write a Python script that simulates the final step of an LLM: the Sampling Layer. We will take a set of raw scores (logits) for potential next words and see how "Temperature" changes what the AI creates.The Experiment:Pythonimport torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. The Vocabulary (Our Periodic Table)
words = ["mat", "couch", "floor", "bed", "moon", "toaster"]

# 2. The Raw Energy Scores (Logits)
# These usually come from the neural network. 
# "mat" (index 0) has the highest score (most likely).
# "toaster" (index 5) has the lowest.
logits = torch.tensor([10.0, 8.0, 7.0, 6.0, 2.0, -5.0])

def apply_temperature(logits, temperature):
    """
    Scales logits by temperature and returns probabilities.
    """
    if temperature == 0:
        # Absolute Zero: Just pick the max (One-Hot encoding)
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
        
    # Apply Thermodynamics formula
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=0)

# 3. Simulation: Varying the Heat
temps = [0.1, 1.0, 2.0, 10.0]
plt.figure(figsize=(15, 4))

for i, T in enumerate(temps):
    probs = apply_temperature(logits, T)
    
    # Plotting
    plt.subplot(1, 4, i+1)
    plt.bar(words, probs.numpy(), color='skyblue')
    plt.title(f"Temp = {T} K")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Collapse the Wave Function (Sample)
# Let's sample 10 times at Temperature 2.0 (High Chaos)
T_simulation = 2.0
probs = apply_temperature(logits, T_simulation)
print(f"Sampling 10 times at Temp {T_simulation}:")

for _ in range(10):
    # torch.multinomial draws a sample based on probabilities
    index = torch.multinomial(probs, 1).item()
    print(words[index], end=", ")
Analyzing the Phase Transitions:Temp 0.1 (Solid): The bar for "mat" will be near 1.0 (100%). The others are near zero. The model is rigid.Temp 1.0 (Liquid): "mat" is still highest (~80%), but "couch" and "floor" have a fighting chance. This is the standard setting for GPT-4.Temp 2.0 (Gas): The bars flatten out. "mat" drops to maybe 40%. "Moon" might jump up to 10%. The model is getting "creative."Temp 10.0 (Plasma): All words have almost equal probability. The model is just guessing randomly. "Toaster" is just as likely as "mat." This is pure noise.Nucleus Sampling (Top-P): Trimming the TailThere is a smarter way to sample than just Temperature. It is called Top-P or Nucleus Sampling.Imagine the probability distribution has a "long tail" of millions of unlikely words. Even with low temperature, if you roll the dice a billion times, eventually you pick a word from the tail.In Top-P, we define a threshold (e.g., $P = 0.9$).We sort the words by probability.We start summing them up from the top: "mat" (15%) + "couch" (10%) + ...As soon as the sum hits 90%, we cut off the rest of the list.We redistribute the 100% probability among only these top candidates.This prevents the model from ever picking "toaster," no matter how chaotic the system gets. It keeps the AI coherent while allowing creativity.TIP: Prompt Engineering Insight.When you use an API (like OpenAI's), you often see temperature and top_p.Modify temperature to control randomness (Logic vs. Poetry).Modify top_p to control vocabulary range (Common words vs. Rare words).Usually, change one or the other, not both.The Context Window: The Observable UniverseAn LLM cannot see forever. It has a Context Window.This is the maximum number of tokens it can hold in its "working memory" (e.g., 4,096 tokens or 128k tokens).If your conversation exceeds this limit, the earliest tokens "fall off the edge of the universe." They are physically deleted from the calculation. The model forgets your name if you told it 5,000 tokens ago.RAG (Retrieval-Augmented Generation) is the hack we use to fix this. We store memories in an external database and "teleport" relevant ones back into the context window just before the model generates an answer.Part III ConclusionWe have covered the creation of Images (Diffusion) and Text (LLMs). We understand the thermodynamics of creativity.But these models are passive. They only speak when spoken to. They do not do anything.In Part IV, we will give them agency. We will turn them into Agents that can use tools, browse the web, and pursue goals.