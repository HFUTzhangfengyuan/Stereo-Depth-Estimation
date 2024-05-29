# %%
from dataset import *
from models.model import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader, random_split
import random

# %%
torch.set_float32_matmul_precision("medium")

# Load data
test = flythings3d("data/flythings3d/", mode="test", downsampling=2)


# Load model
model = StereoNet(in_channels=3, K=2, D=256)
# model = StereoNet.load_from_checkpoint("epoch=9-step=223900.ckpt", in_channels=3, K=2)



checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
trainer = L.Trainer(
    max_epochs=1, accumulate_grad_batches=1, callbacks=[checkpoint_callback]
)

# 加载参数
model.load_state_dict(torch.load("./Model-save/cre_fly_SEAttention.ckpt"))

# 训练结束后，会将参数保存到model.ckpt文件中，如果后面需要测试，只用把训练代码部分注释，然后加载模型参数到model中，然后执行下面三行代码
trainer.test(model=model, dataloaders=DataLoader(test, batch_size=1, num_workers=23))
loss = model.get_meanloss()
print(f"Test mean_loss: {loss:.6f}")
