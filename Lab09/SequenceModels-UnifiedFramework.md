# Evolution of Deep Sequential Models

### Notation

* ($A$): state transform
* ($B$): input projection
* ($C$): output projection
* ($h_t$): exposed hidden state
* ($m_t$): (long-term) vector memory
* ($M_t$): matrix memory
* ($\mathbf{M}$): global sequence operator

---


## 1. Classical RNN

Single-state nonlinear dynamics ($f$ is nonlinear):
* $h_t = f(Ah_{t-1} + Bx_t)$
* $y_t = C h_t$

> Capacity: O(d). Short-term compression only.
---

## 2. Gated RNN (LSTM / GRU)

Vector memory controls retention:
* $m_t = g(Ah_{t-1}, Bx_t, m_{t-1})$
* $h_t = f(Ah_{t-1}, Bx_t, m_t)$
* $y_t = C h_t$

---

## 3. xLSTM - Explicit Matrix Memory

Matrix memory via outer products
* $M_t = g_{exp}(Ah_{t-1}, Bx_t, M_{t-1})$
* $h_t = f(Ah_{t-1}, Bx_t, M_t)$
* $y_t = C h_t$

> Memory type:
> * Content-addressable memory field
> * Explicit associative memory via outer products
> * High capacity ($O(d^2)$)

---

## 4. SSM (LTI/LTV) — Linear State Evolution

### A. LTI-SSM

* $h_t = A h_{t-1} + B x_t$
* $y_t = C h_t$

### B. LTV-SSM

* $h_t = A_t h_{t-1} + B_t x_t$
* $y_t = C_t h_t$

> Memory type:
> * Linear, input-gated
> * State ($h_t$) is an expanded compressed history => O(d × N) effective capacity
> * The shift from LTI to LTV introduces selective retention/forgetting
---

# 5. SSM Global View (Parallel)

### A. LTI → Convolution

* $y = x * \bar{K}$
* $\bar{K} = (CB, CA B, CA^2B, \dots)$

### B. LTV → SSD (Mamba-2)

* $y = \mathbf{M} x$
* $\mathbf{M} = L \circ (C B^\top)$

> Components:
> * ($C B^\top$): Input–Output Interaction Matrix
> * ($L$): Structural mask from dynamics ($A$)
> * ($\circ$): Hadamard product
> * ($M$): Global sequence operator (defines the model’s memory structure)

---

# Evolution of the Hidden State

### RNN

* Hidden state = nonlinear compression of history.

### Gated RNN

* Splits memory into protected long-term memory ($m_t$) and exposed state ($h_t$).

### xLSTM

* Upgrades memory to matrix form ($M_t$) for high-capacity storage.

### SSM

* Re-unifies state and memory into a single expanded state..
* Uses selective linear dynamics instead of explicit gates.
* Implicit high-order memory through the structure of $M$.
