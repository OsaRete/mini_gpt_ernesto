# Mini GPT – Character-level Language Model (Python + NumPy)

Hi! I'm **Ernesto Cedeño**, a Software Engineering student, and this is my own **mini GPT-style language model**, built from scratch using **Python and NumPy only**.

The goal of this project is educational: to understand how an autoregressive language model works internally, without relying on high-level deep learning frameworks.

---

## 🚀 What this project does

- Trains a **character-level language model** from plain text (`data/input.txt`)
- Uses:
  - Character vocabulary
  - Embeddings
  - Fixed context window (block size)
  - A small **MLP (hidden layer with tanh)**
  - Cross-entropy loss and gradient descent
- Generates new text **character by character**, in the style of the training data

This is not meant to compete with GPT-4, of course 😄  
But it helps to understand the core ideas behind large language models.

---

## 🧠 Model: `mini_gpt_v2.py`

The file `mini_gpt_v2.py` implements:

- `MiniGPTMLP` class:
  - Builds a vocabulary from the training text
  - Creates (context, next_char) pairs using a sliding window
  - Learns embeddings for each character
  - Concatenates embeddings → passes them through an MLP
  - Predicts the probability distribution over the next character
- Training loop:
  - Mini-batch gradient descent
  - Cross-entropy loss
  - Periodic loss logging
- Text generation:
  - Starts from an initial text like `"Hola"`
  - Uses the last `block_size` characters as context
  - Samples the next character from the model's probabilities
  - Repeats autoregressively
- Weight saving:
  - Saves trained parameters to `mini_gpt_v2_weights.npz`

---

## 📁 Project structure

```text
mini_gpt_ernesto/
│
├─ data/
│   └─ input.txt              # Training text dataset
│
├─ mini_gpt_v2.py             # Model + training + generation
└─ README.md                  # Project documentation