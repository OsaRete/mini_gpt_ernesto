import numpy as np
from pathlib import Path


class MiniGPTMLP:
    """
    Mini modelo de lenguaje tipo GPT a nivel de caracteres, usando:
    - contexto de varios caracteres (ventana fija)
    - embeddings
    - MLP (capa oculta con tanh)
    - entrenamiento por entropía cruzada

    No es un Transformer, pero ya es un modelo neuronal autoregresivo real.
    """

    def __init__(self, text, block_size=8, emb_dim=16, hidden_dim=64, lr=1e-2, seed=42):
        np.random.seed(seed)

        self.block_size = block_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # --- 1. Vocabulario de caracteres ---
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # --- 2. Codificar el texto en índices ---
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)

        # --- 3. Crear dataset de pares (contexto, siguiente carácter) ---
        xs, ys = [], []
        for i in range(block_size, len(data)):
            context = data[i - block_size:i]   # (block_size,)
            target = data[i]                   # escalar
            xs.append(context)
            ys.append(target)

        self.xs = np.array(xs, dtype=np.int64)  # (N, block_size)
        self.ys = np.array(ys, dtype=np.int64)  # (N,)

        # --- 4. Inicializar parámetros del modelo ---
        # Embeddings: tabla de vectores para cada carácter
        self.C = 0.01 * np.random.randn(self.vocab_size, emb_dim)

        # MLP: [embeddings concatenados] -> hidden -> logits
        self.W1 = 0.01 * np.random.randn(block_size * emb_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = 0.01 * np.random.randn(hidden_dim, self.vocab_size)
        self.b2 = np.zeros(self.vocab_size)

    @staticmethod
    def _softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, x_batch):
        """
        x_batch: (B, block_size) con índices de caracteres
        Devuelve:
        - probs: (B, vocab_size)
        - cache: valores intermedios para backprop
        """
        # Embeddings: (B, block_size, emb_dim)
        emb = self.C[x_batch]

        # Aplanar contexto: (B, block_size * emb_dim)
        emb_flat = emb.reshape(emb.shape[0], -1)

        # Capa oculta
        h_pre = emb_flat @ self.W1 + self.b1  # (B, hidden_dim)
        h = np.tanh(h_pre)

        # Logits salida
        logits = h @ self.W2 + self.b2        # (B, vocab_size)

        probs = self._softmax(logits)

        cache = (emb, emb_flat, h_pre, h, logits, x_batch)
        return probs, cache

    def _backward(self, probs, cache, y_batch):
        """
        Calcula gradientes de todos los parámetros dado el batch.
        """
        emb, emb_flat, h_pre, h, logits, x_batch = cache
        B = y_batch.shape[0]

        # Gradiente de la pérdida w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(B), y_batch] -= 1
        grad_logits /= B  # promediamos

        # Gradientes de W2 y b2
        dW2 = h.T @ grad_logits              # (hidden_dim, vocab_size)
        db2 = grad_logits.sum(axis=0)        # (vocab_size,)

        # Gradiente hacia la capa oculta
        dh = grad_logits @ self.W2.T         # (B, hidden_dim)
        dh_pre = (1 - np.tanh(h_pre) ** 2) * dh  # derivada de tanh

        # Gradientes de W1 y b1
        dW1 = emb_flat.T @ dh_pre            # (block_size*emb_dim, hidden_dim)
        db1 = dh_pre.sum(axis=0)             # (hidden_dim,)

        # Gradiente hacia embeddings aplanados
        demb_flat = dh_pre @ self.W1.T       # (B, block_size*emb_dim)
        demb = demb_flat.reshape(emb.shape)  # (B, block_size, emb_dim)

        # Gradiente de la tabla de embeddings C
        dC = np.zeros_like(self.C)
        # Para cada posición (B, block_size) acumulamos
        np.add.at(dC, x_batch, demb)

        return dC, dW1, db1, dW2, db2

    def train(self, steps=5000, batch_size=32, print_every=500):
        for step in range(1, steps + 1):
            # Elegir un mini-batch aleatorio
            idx = np.random.randint(0, len(self.xs), size=(batch_size,))
            x_batch = self.xs[idx]  # (B, block_size)
            y_batch = self.ys[idx]  # (B,)

            # Forward
            probs, cache = self._forward(x_batch)

            # Pérdida
            loss = -np.log(probs[np.arange(batch_size), y_batch] + 1e-9).mean()

            # Backward
            dC, dW1, db1, dW2, db2 = self._backward(probs, cache, y_batch)

            # Actualizar parámetros
            self.C  -= self.lr * dC
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if step % print_every == 0:
                print(f"Step {step}/{steps} - Loss: {loss:.4f}")

    def generate(self, start_text, length=200):
        """
        Genera texto carácter por carácter usando contexto de longitud block_size.
        """
        # Si el texto inicial es más corto que block_size, lo rellenamos al inicio
        if len(start_text) < self.block_size:
            start_text = (" " * (self.block_size - len(start_text))) + start_text
        else:
            start_text = start_text[-self.block_size:]

        context = [self.stoi.get(c, 0) for c in start_text]
        out_chars = list(start_text)

        for _ in range(length):
            x = np.array([context], dtype=np.int64)  # (1, block_size)
            probs, _ = self._forward(x)
            probs = probs[0]

            idx_next = np.random.choice(self.vocab_size, p=probs)
            ch_next = self.itos[idx_next]
            out_chars.append(ch_next)

            # Actualizar contexto: deslizamos la ventana
            context = context[1:] + [idx_next]

        return "".join(out_chars)

    def save(self, path="mini_gpt_v2_weights.npz"):
        np.savez(
            path,
            C=self.C,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            chars=np.array(self.chars),
            block_size=self.block_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
        )
        print(f"Pesos guardados en {path}")


def load_text(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {path}. "
            f"Crea un archivo de texto en '{path}' con tu dataset."
        )
    return path.read_text(encoding="utf-8")


def main():
    text = load_text("data/input.txt")
    print(f"Texto cargado con {len(text)} caracteres.")
    print(f"Caracteres únicos (vocab_size): {len(set(text))}")

    model = MiniGPTMLP(
        text,
        block_size=8,
        emb_dim=16,
        hidden_dim=64,
        lr=1e-2,
    )

    print("Entrenando modelo (v2, MLP con contexto)...")
    model.train(steps=4000, batch_size=64, print_every=400)

    print("\n=== TEXTO GENERADO (v2) ===")
    sample = model.generate(start_text="Hola", length=400)
    print(sample)

    model.save("mini_gpt_v2_weights.npz")


if __name__ == "__main__":
    main()


  