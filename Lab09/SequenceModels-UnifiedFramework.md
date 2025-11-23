# Evolution of Deep Sequential Models

### Notation

* ($A$): state transform
* ($B$): input projection
* ($C$): output projection
* ($h_t$): exposed hidden state
* ($m_t$): vector memory
* ($M_t$): matrix memory
* ($\mathbf{M}$): global sequence operator

---


## 1. Classical RNN

Single-state nonlinear dynamics ($f$ is nonlinear):
* $h_t = f(Ah_{t-1} + Bx_t)$
* $y_t = C h_t$

> Capacity: $O(d)$. Short-term compression only.
---

## 2. Gated RNN (LSTM / GRU)

Vector memory controls retention:
* $m_t = g(Ah_{t-1}, Bx_t, m_{t-1})$
* $h_t = f(Ah_{t-1}, Bx_t, m_t)$
* $y_t = C h_t$

> Capacity: $O(d)$. Uses gated memory $m_t$ to control retention, enabling long-term memory.
---

## 3. xLSTM - Explicit Matrix Memory

Matrix memory via outer products
* $M_t = g_{exp}(Ah_{t-1}, Bx_t, M_{t-1})$
* $h_t = f(Ah_{t-1}, Bx_t, M_t)$
* $y_t = C h_t$

> Memory type:
> * Content-addressable memory field
> * Explicit associative memory via outer products
> * High capacity $(O(d^2))$

---
## 4. S4 (LTI-SSM) — The Convolutional Era
Fixed Linear Dynamics (Time-Invariant):
* Recurrent:
  * $h_t = A h_{t-1} + B x_t$
  * $y_t = C h_t$
* Global (Convolution):
  * $y = x * \bar{K}$
  * $\bar{K} = (CB, CA B, CA^2B, \dots)$

> Mechanism:
> * Parameters ($A, B, C$) are fixed for the whole sequence.
> * $A$ is typically derived from a continuous system, but acts as a fixed linear operator per step.
> * Allows transforming from the time domain to the frequency domain via FFT.
> * Cannot selectively forget/remember (no context-dependence).

---

## 5. Mamba (Selective SSM) — The Selection Era
* Recurrent:
  * $h_t = A_t h_{t-1} + B_t x_t$
  * $y_t = C_t h_t$
* Global (Parallel Scan):
  * $h = \text{scan}(A_t, B_t, x)$
  * $y = C_t \cdot h$

> Mechanism:
> * $A_t$ and $B_t$ are functions of the current input $x_t$.
> * State ($h_t$) is an expanded compressed history $(O(d \times N))$.
> * Replaces explicit gates with selective decay/update rates.
> * Hardware-aware Parallel Associative Scan 

---

## 6. Mamba-2 (SSD) — The Duality Era
Structured State Space Duality:
* Recurrent:
    *   $h_t = A_t h_{t-1} + B_t x_t$
    *   $y_t = C_t h_t$
*   Global (Matrix):
    *   $y = \mathbf{M} x$
    *   $\mathbf{M} = L \circ (C B^\top)$
> Mechanism:
> *   Duality: Proves SSMs are dual to Linear Attention.
> *   Components:
>     *   ($C B^\top$): Input–Output Interaction (Attention-like).
>     *   ($L$): Mask matrix derived from the cumulative decay of $A_t$.
>     *   ($\circ$): Hadamard product
> *   Memory type:
>     *   State ($h_t$) is an expanded compressed history ($O(d \times N)$).
>     *   During training (Global), the memory is implicit in the interaction matrix.
>     *   During inference (Recurrent), it collapses back to the fixed-size state $h_t$.
> *   Compute: Block-Decomposed Matrix Multiplication.


---

# Evolution of the Hidden State

### RNN

* Hidden state = nonlinear compression of history.

### Gated RNN

* Splits memory into protected long-term memory ($m_t$) and exposed state ($h_t$).

### xLSTM

* Upgrades memory to matrix form ($M_t$) for high-capacity storage.

### S4 (SSM)
*   Re-unifies state and memory ($h_t$).
*   Uses fixed linear dynamics for long-range context window (via convolution).

### Mamba / Mamba-2
*   Introduces Selection: Dynamics change per token.
*   Mamba-1: Uses Scan to compute selective recurrence.
*   Mamba-2: Uses Matrix Duality ($\mathbf{M}$) to compute selective recurrence as an interaction matrix.
