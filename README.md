<div align="center">
  <h1>Radon Averaging</h1>
  <h3>A Practical Approach for Designing Rotation-Invariant Models</h3>
  
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&style=flat-square" alt="Python Badge"/>
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
  <img src="./radon_rotation.gif" width="700px"/>
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

## Method Overview

Radon Averaging achieves rotation invariance by:
1. **Radon Transform** (ℛ): Converts images to sinograms where rotation → circular shift
2. **Averaging over Discrete Rotations**: Eliminates boundary artifacts via group averaging
3. **Standard CNN Backbone**: No architectural changes required
```math
RA_G^Φ(I) = \frac{1}{|G|} \sum_{g \in G} (Φ \circ π(g) \circ ℛ)(I)
```
