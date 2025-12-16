<div align="center">
  <h1>Radon Averaging</h1>
  <h3>A Practical Approach for Designing Rotation-Invariant Models</h3>
  
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=flat-square" alt="Python Badge"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&style=flat-square" alt="PyTorch Badge"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://doi.org/10.1016/j.engappai.2025.113299">
    <img src="https://img.shields.io/badge/EAAI%202026-Published-success?style=flat-square" alt="EAAI Badge"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://www.elsevier.com/">
    <img src="https://img.shields.io/badge/Elsevier-Journal-orange?style=flat-square" alt="Elsevier Badge"/>
  </a>
  <br/><br/>
  
  <!-- Radon Transform Animation -->
  <img src="./gif_for_readme.gif" width="700px"/>
</div>

---

## Engineering Applications of Artificial Intelligence (EAAI 2026)
### Pytorch Implementation

This repository contains a pytorch implementation of **Radon Averaging (RA)** from the paper:

> **Radon Averaging: A practical approach for designing rotation-invariant models**  
> Jangwon Kim, Sanghyun Ryoo, Jiwon Kim, Junkee Hong, Soohee Han  
> *Engineering Applications of Artificial Intelligence*, Volume 164, 2026

## 📄 Paper Link
> **DOI:** https://doi.org/10.1016/j.engappai.2025.113299  
> **Journal:** Engineering Applications of Artificial Intelligence

---

## Radon Averaging

Radon Averaging achieves rotation invariance by:
1. **Radon Transform** (ℛ): Converts images ($I$) to sinograms, where an rotation corresponds ($$g$$) to a circular shift.
2. **Averaging over Discrete Rotations** ($$G$$): Eliminates boundary artifacts via group averaging
3. **Standard CNN Backbone** ($$Φ$$): No architectural changes required
```math
RA_G^Φ(I) = \frac{1}{|G|} \sum_{g \in G} (Φ \circ π(g) \circ ℛ)($$)
```
---

## Advantages
- **Plug-and-play**: works with standard (pretrained) CNN backbones (no architectural changes).
- **Rotation invariance in practice**: stable representations under image rotations.
- **Reduces boundary artifacts**: group averaging mitigates Radon transform edge effects.
---

## Citation Example
```
@article{kim2026radonaveraging,
  title   = {Radon Averaging: A practical approach for designing rotation-invariant models},
  author  = {Kim, Jangwon and Ryoo, Sanghyun and Kim, Jiwon and Hong, Junkee and Han, Soohee},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {164},
  pages   = {113299},
  year    = {2026},
  doi     = {10.1016/j.engappai.2025.113299}
}
```
