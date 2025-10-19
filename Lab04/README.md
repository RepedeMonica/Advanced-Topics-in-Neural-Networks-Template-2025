## Lab 4

***


Lab Notebook

1. [Using C++ Modules in PyTorch](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/advanced_pytorch/UsingCppModules.ipynb)


<details><summary>Bonus points</summary>
You will get bonus points if you implement the torchvision transforms from "Complex Yet Simple Training Pipeline" in C++ and submit them until Lab 6.
</details>


***

For self-study (for students who want to pass):
* Foundational CNN papers:
  * AlexNet: https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf
  * ResNet: https://arxiv.org/abs/1512.03385
* Advanced optimizers:
  * SAM Optimizer: https://github.com/davda54/sam
  * Muon Optimizer: https://kellerjordan.github.io/posts/muon/
* Hyperparameter tuning / experiment tracking:
  * Tensorboard: https://pytorch.org/docs/stable/tensorboard.html
  * Weights and Biases: https://docs.wandb.ai/guides/integrations/pytorch
* Parallelism: https://docs.pytorch.org/tutorials/beginner/dist_overview.html
  * Tensor parallelism: https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
  * Distributed Data Parallel: https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html

***


Advanced (for students who want to learn more):
* C++ & CUDA:
    * Introduction to CUDA: https://developer.nvidia.com/blog/even-easier-introduction-cuda
    * Optimizing preprocessing pipelines with C++ modules: https://medium.com/data-science/how-to-optimize-your-dl-data-input-pipeline-with-a-custom-pytorch-operator-7f8ea2da5206
* SAM Optimizer:
  * Sharpness-Aware Minimization for Efficiently Improving Generalization: https://arxiv.org/abs/2010.01412
* Muon Optimizer:
  * Pytorch implementation: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
  * Muon is Scalable for LLM Training: https://arxiv.org/pdf/2502.16982
  * Use the Muon implementation from timm if you have >2D weight matrices in your network.
* Parallelism tutorials:
  * https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
  * https://huggingface.co/blog/huseinzol05/tensor-parallelism
  * https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html
