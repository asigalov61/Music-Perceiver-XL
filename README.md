# Music Perceiver XL
## A hierarchical, Perceiver‑augmented transformer for long‑form symbolic‑music modeling with bar‑level and global latent memory

<img width="1024" height="1024" alt="Music-Perceiver-XL-Logo" src="https://github.com/user-attachments/assets/e399ae78-fb52-4bbd-a754-b82f1eacedb9" />

***

### **Abstract**
 
> A **hierarchical, memory‑augmented autoregressive architecture** for symbolic‑music modeling, implemented in **PyTorch / x‑transformers 2.3.1**, that introduces **bar‑level** and **global‑level Perceiver memories**, dynamic bar segmentation, and fully streaming generation with KV‑cache support.

---

### **Concise Takeaway**  
**Music Perceiver XL** extends a transformer decoder with **two stacked PerceiverMemory modules** that compress token sequences into **bar‑structured latent summaries** and then into **global musical context**, enabling long‑range coherence, efficient conditioning, and generation beyond fixed sequence limits.

---

### **Comprehensive Abstract**  
**Music Perceiver XL** is a hierarchical autoregressive model designed for long‑form symbolic‑music generation. The architecture augments a standard transformer decoder with **multi‑scale Perceiver memories**, enabling efficient compression of musical structure while preserving global coherence. The model first embeds incoming tokens and identifies **bar boundaries** using a configurable token‑ID mask. Tokens are dynamically grouped into bars through cumulative indexing, sorted into contiguous bar segments, padded, and masked. Each bar is encoded by a **PerceiverMemory** module that performs cross‑attention from a fixed set of learnable bar latents to the padded bar tokens, followed by self‑attention and feed‑forward refinement. Bar latents are enriched with **bar‑index embeddings** to preserve musical order.

A second **global PerceiverMemory** aggregates all bar‑level latents into a compact global context representation. The resulting hierarchical memory—global latents followed by bar latents—is projected into the decoder dimension and concatenated into a unified **cross‑attention context**. A dynamic mask is constructed to ensure correct handling of padded bars and to work around x‑transformers v2.3.1’s context‑padding behavior.

The decoder is a **TransformerWrapper** with rotary positional embeddings, flash attention, and full **KV‑cache support**, enabling efficient streaming generation and inference beyond the nominal maximum sequence length. The model supports optional continuous prediction heads and exposes a clean interface for cache clearing and hierarchical memory construction.

Overall, Music Perceiver XL provides a **scalable, structure‑aware, and computationally efficient** approach to modeling long musical sequences by combining transformer decoding with multi‑level Perceiver‑style latent compression, enabling rich global musical structure and long‑range dependencies to be captured within a tractable latent hierarchy.


***

## Install

```sh
!git clone https://github.com/asigalov61/Music-Perceiver-XL

!pip install tqdm
!pip install einx
!pip install torch
!pip install einops
!pip install torch-summary
!pip install matplotlib
!pip install scikit-learn
!pip install numpy
```

***

### Project Los Angeles
### Tegridy Code 2026
