# Latent Knowledge Scalpel: Precise and Massive Knowledge Editing for Large Language Models

## Abstract
Large Language Models (LLMs) often retain inaccurate or outdated information from pre-training, leading to incorrect predictions or biased outputs during inference. While existing model editing methods can address this challenge, they struggle with editing large amounts of factual information simultaneously and may compromise the general capabilities of the models. In this paper, our empirical study demonstrates that it is feasible to edit the internal representations of LLMs and replace the entities in a manner similar to editing natural language inputs. Based on this insight, we introduce the Latent Knowledge Scalpel (LKS), an LLM editor that manipulates the latent knowledge of specific entities via a lightweight hypernetwork to enable precise and large-scale editing. Experiments conducted on Llama-2 and Mistral show even with the number of simultaneous edits reaching 10,000, LKS effectively performs knowledge editing while preserving the general abilities of the edited LLMs. 

## How to Cite
```
 @misc{liu2025latentknowledgescalpelprecise,
      title={Latent Knowledge Scalpel: Precise and Massive Knowledge Editing for Large Language Models}, 
      author={Xin Liu and Qiyang Song and Shaowen Xu and Kerou Zhou and Wenbo Jiang and Xiaoqi Jia and Weijuan Zhang and Heqing Huang and Yakai Li},
      year={2025},
      eprint={2508.03741},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.03741}, 
}
```
