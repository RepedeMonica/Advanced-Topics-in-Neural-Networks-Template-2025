## Lab 1

***

PyTorch Recap:
* [01. Custom Dataset](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/01_custom_dataset.ipynb)
* [02. DataLoader](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/02_dataloader.ipynb)
* [03. Simple Training](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/03_simple_training.ipynb)
* [04. Optimizers](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/04_optimizers.ipynb) 
* [05. Learning Rate Schedulers](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/05_lr_schedulers.ipynb) - you have to work through this at home
* [06. Simple Data Augmentation](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/beginner_pytorch/06_data_augmentation.ipynb) - you have to run this during the lab for visualizations

The exercises from Notebook 4, 5, 6 should be done at home, before/after the lab. They are a good way of learning DL and PyTorch. 
<details><summary>Bonus points</summary>
You will get bonus points if you do at least 4 out of 5 exercises in Notebook 4 and submit them until Lab 3.
For exercise 1, you also need to record the measurements and explain what did you change and why.
</details>

***

Lab Notebooks

1. [Complex Yet Simple Training Pipeline](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/advanced_pytorch/ComplexYetSimpleTrainingPipeline.ipynb)
2. [Inference Optimization and TTA](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/resources/advanced_pytorch/InferenceOptimizationAndTTA.ipynb)

<details><summary>Bonus points</summary>
You will get bonus points if you do all 4 exercises from "Complex Yet Simple Training Pipeline" and submit them until Lab 5.
</details>

***

For self-study (for students who want to pass):
* [Neural Networks (chapter 1 - chapter 4)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation) - last chance before Homework 1
* Dataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
* DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
* TorchVision transforms getting started: https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
* TorchVision examples: https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html
***


Advanced (for students who want to learn more):
* Considering following the [roadmap](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/foundations/roadmap.md) at your own pace. Do the exercises in each notebook.
* `pin_memory` & `non_blocking=True`:
   * Pinning memory in DataLoaders: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
   * How does pinned memory actually work: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ 
* Data Augmentation for CV:
  * [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
  * [Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
  * [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
