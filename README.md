<img src="https://cdn.svgporn.com/logos/pytorch.svg" align="right" width="15%"/>

# Transformer
Learn Transformer model ([Attention is all you need, Google Brain, 2017](https://arxiv.org/abs/1706.03762)) from implementation code written by [@hyunwoongko](https://github.com/hyunwoongko/transformer) in 2021.

[![Python](https://img.shields.io/badge/Python-3.8.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red?logo=pytorch)](https://pytorch.org/)
[![VSCode](https://img.shields.io/badge/VSCode-1.60.2-white?logo=visualstudiocode)](https://code.visualstudio.com/)

## Results

![graph](results/graph.png)

- Min train loss = 2.7348
- Min validation loss = 3.2860
- Max blue score = 24.3375

| Model | Dataset | BLEU Score |
|:---:|:---:|:---:|
| Hyunwoong Ko's | Multi30K EN-DE | 26.4 |
| My Implementation | Multi30K EN-DE | 24.3 |

## Reference
- [Attention is All You Need, 2017 - Google Brain](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Multi-Headed Attention implementation *PyTorch*](https://nn.labml.ai/transformers/mha.html)
- [Transformers from scratch *blog*](http://peterbloem.nl/blog/transformers)
- :hugs: [Hugging Face course](https://huggingface.co/course/chapter1)
- [How Transformers work in deep learning and NLP: an intuitive introduction ](https://theaisummer.com/transformer/)
- [Why multi-head self attention works: math, intuitions and 10+1 hidden insights](https://theaisummer.com/self-attention/)

