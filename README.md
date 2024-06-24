

FSL中的数据加载与标准分类略有不同，因为我们以少量分类任务的形式采样批量实例。
- [TaskSampler](easyfsl/samplers/task_sampler.py): an extension of the standard PyTorch Sampler object, to sample batches in the shape of few-shot classification tasks
- [FewShotDataset](easyfsl/datasets/few_shot_dataset.py): an abstract class to standardize the interface of any dataset you'd like to use
- [EasySet](easyfsl/datasets/easy_set.py): a ready-to-use FewShotDataset object to handle datasets of images with a class-wise directory split
- [WrapFewShotDataset](easyfsl/datasets/wrap_few_shot_dataset.py): a wrapper to transform any dataset into a FewShotDataset object
- [FeaturesDataset](easyfsl/datasets/features_dataset.py): a dataset to handle pre-extracted features
- [SupportSetFolder](easyfsl/datasets/support_set_folder.py): a dataset to handle support sets stored in a directory

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

