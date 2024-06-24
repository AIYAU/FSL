

FSL中的数据加载与标准分类略有不同，因为我们以少量分类任务的形式采样批量实例。
- [TaskSampler](easyfsl/samplers/task_sampler.py): 标准PyTorch Sampler对象的扩展，以少量分类任务的形式对批次进行采样
- [FewShotDataset](easyfsl/datasets/few_shot_dataset.py): 一个抽象类，用于标准化您想要使用的任何数据集的接口
- [EasySet](easyfsl/datasets/easy_set.py): 一个随时可用的FewShotDataset对象，用于处理具有类分类的目录分割的图像数据集
- [WrapFewShotDataset](easyfsl/datasets/wrap_few_shot_dataset.py): 将任何数据集转换为一个FewShotDataset对象的包装器
- [FeaturesDataset](easyfsl/datasets/features_dataset.py): 处理预提取特征的数据集
- [SupportSetFolder](easyfsl/datasets/support_set_folder.py): 用于处理存储在目录中的支持集的数据集
### Datasets to test your model

**[CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)**

```python
from easyfsl.datasets import CUB

train_set = CUB(split="train", training=True)
test_set = CUB(split="test", training=False)
```

**[tieredImageNet](https://paperswithcode.com/dataset/tieredimagenet)**

```python
from easyfsl.datasets import TieredImageNet

train_set = TieredImageNet(split="train", training=True)
test_set = TieredImageNet(split="test", training=False)
```

**[miniImageNet](https://paperswithcode.com/dataset/miniimagenet)**

```python
from easyfsl.datasets import MiniImageNet

train_set = MiniImageNet(root="where/imagenet/is", split="train", training=True)
test_set = MiniImageNet(root="where/imagenet/is", split="test", training=False)
```

**[Danish Fungi](https://paperswithcode.com/paper/danish-fungi-2020-not-just-another-image)**

```python
from easyfsl.datasets import DanishFungi

dataset = DanishFungi(root="where/fungi/is")
```

## QuickStart


1. Install the package: ```pip install easyfsl``` or simply fork the repository.
   
2. [Download your data](#datasets-to-test-your-model).

3. Design your training and evaluation scripts. You can use our example notebooks for 
[episodic training](notebook/05episodic_training.ipynb) 
or [classical training](notebook/02classical_training_with_PrototypicalNetworks.ipynb).

