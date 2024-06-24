# 导包
import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

#设置随机种子
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 加载鸟类数据集
from easyfsl.datasets import CUB
from torch.utils.data import DataLoader

batch_size = 128
n_workers = 12 # 如果你是Linux那么你可以随便设置，如果你是windows，那你洗洗睡吧

train_set = CUB(split="train", training=True)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=n_workers,
    pin_memory=True,
    shuffle=True,
)

from easyfsl.modules import resnet12

DEVICE = "cuda"

model = resnet12( # 定义主干网络
    use_fc=True,
    num_classes=len(set(train_set.get_labels())),
).to(DEVICE)


from easyfsl.methods import PrototypicalNetworks # 使用原型网络进行
from easyfsl.samplers import TaskSampler

n_way = 5
n_shot = 5
n_query = 10
n_validation_tasks = 500

# 这波定义验证集
val_set = CUB(split="val", training=False)
val_sampler = TaskSampler(
    val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)

from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


LOSS_FUNCTION = nn.CrossEntropyLoss()

n_epochs = 200
scheduler_milestones = [150, 180]
scheduler_gamma = 0.1
learning_rate = 1e-01
tb_logs_dir = Path("./logs") # 设置tensorbroad的保存目录

train_optimizer = SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
)
train_scheduler = MultiStepLR(
    train_optimizer,
    milestones=scheduler_milestones,
    gamma=scheduler_gamma,
)

tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

# 训练一个epoch，这里是训练主干网络及特征提取器
def training_epoch(model_: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train: # 遍历数据集
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            loss = LOSS_FUNCTION(model_(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)


from easyfsl.utils import evaluate


best_state = model.state_dict() # 定义最佳模型
best_validation_accuracy = 0.0
validation_frequency = 10
for epoch in range(n_epochs): # 开始训练
    print(f"Epoch {epoch}")
    average_loss = training_epoch(model, train_loader, train_optimizer) # 传入定义的模型和数据集优化器配置等

    if epoch % validation_frequency == validation_frequency - 1: # 如果达到预设的验证要求就验证模型

        # We use this very convenient method from EasyFSL's ResNet to specify
        # that the model shouldn't use its last fully connected layer during validation.
        model.set_use_fc(False)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )
        model.set_use_fc(True)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

    tb_writer.add_scalar("Train/loss", average_loss, epoch)

    # Warn the scheduler that we did an epoch
    # so it knows when to decrease the learning rate
    train_scheduler.step()


model.load_state_dict(best_state)# 加载最优权重
n_test_tasks = 1000

test_set = CUB(split="test", training=False)
test_sampler = TaskSampler(
    test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
)
test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

model.set_use_fc(False)

accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
print(f"Average accuracy : {(100 * accuracy):.2f} %")