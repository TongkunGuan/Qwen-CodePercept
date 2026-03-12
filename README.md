# CodePercept: Code-Grounded Visual STEM Perception for MLLMs
> 
> **Official repository for the CVPR 2026 paper: "CodePercept: Code-Grounded Visual STEM Perception for MLLMs"**.
> 
> Note: Code and weights will be available soon.
> 
[paper link](http://arxiv.org/abs/2603.10757)
## 💡 About

When Multimodal Large Language Models (MLLMs) fail at Science, Technology, Engineering, and Mathematics (STEM) visual reasoning, a fundamental question arises: is it due to perceptual deficiencies or reasoning limitations?.

Through a systematic scaling analysis, we reveal that **perception is the true bottleneck limiting current STEM visual reasoning**. Motivated by this insight, CodePercept introduces a paradigm shift: using **executable code as a powerful perceptual medium**. Because executable code provides precise semantics that naturally align with the structured nature of STEM visuals, it overcomes the hallucinations and "descriptive aphasia" inherent in natural language.

## 🚀 Key Contributions

* **Code-as-Perception Paradigm**: We establish that scaling perception consistently outperforms scaling reasoning in STEM domains.

* **ICC-1M Dataset**: We construct a large-scale dataset comprising over 1 million high-quality STEM Image-Caption-Code triplets. Data is synthesized via three pipelines: Image Reproduce, Image Diversity, and Solid Geometry Synthesis.

* **Novel Training Tasks**:
* **Code-Grounded Caption Generation**: Treats executable code as ground truth for image captions, eliminating hallucinations found in existing knowledge distillation methods.
* **STEM Image-to-Code Translation**: Prompts models to generate reconstruction code, mitigating the ambiguity of natural language descriptions.

* **STEM2Code-Eval Benchmark**: A manually annotated benchmark of 1,000 images that evaluates visual perception through deterministic, executable Python code generation, moving beyond traditional problem-solving accuracy proxies.



## 🧠 Methodology & Training
<img width="1073" height="911" alt="image" src="https://github.com/user-attachments/assets/9d63ad5d-dabc-4a34-a3b1-8a72fef58504" />


Our model is built upon the Qwen3-VL architecture and trained via a two-stage paradigm:

1. **Supervised Finetuning (CodePercept-S1)**: Jointly optimizes image captioning and image-to-code translation using our curated ICC-1M triplets.
2. **Reinforcement Learning (CodePercept-R1)**: We employ Group Relative Policy Optimization (GRPO) specifically targeting code generation. The reward mechanism enforces:
* **Format Reward**: Ensuring valid Python syntax.
* **Content Reward**: Combining execution success, code-level semantic equivalence (via GPT-4o), and image-level visual similarity.





## 📊 Evaluation & Results

CodePercept demonstrates state-of-the-art performance across multiple evaluations:

* **STEM Reasoning Benchmarks**: CodePercept-8B-S1 outperforms super-large models like Qwen2.5-VL-72B and approaches the performance of frontier models when paired with a strong LLM solver.
<img width="698" height="467" alt="image" src="https://github.com/user-attachments/assets/27b3adeb-565e-42b7-a059-a46221e1a828" />


* **STEM2Code-Eval**: On our novel verifiable perception benchmark, CodePercept-8B-R1 achieves an average score of 63.56, significantly surpassing Qwen3-VL-8B-Instruct (+16.19) and even super-large models like Qwen3-VL-Plus.
<img width="765" height="544" alt="image" src="https://github.com/user-attachments/assets/518b46f8-d259-4321-9b50-eec54bbff333" />



## ⚙️ Quick Start

*(Installation instructions, environment setup, and inference scripts will be updated upon code release.)*

## 📖 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{codepercept2026,
  title={CodePercept: Code-Grounded Visual STEM Perception for MLLMs},
  author={Tongkun Guan, Zhibo Yang, Jianqiang Wan, Mingkun Yang, Zhengtao Guo, Zijian Hu, Ruilin Luo, Ruize Chen, Songtao Jiang, Peng Wang, Wei Shen, Junyang Lin, Xiaokang Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

```
