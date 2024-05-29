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
train = flythings3d("data/flythings3d/", mode="train", downsampling=2)
val = flythings3d("data/flythings3d/", mode="val", downsampling=2)
# test = flythings3d("data/flythings3d/", mode="test", downsampling=2)


# Load model
model = StereoNet(in_channels=3, K=2, D=256)
# model = StereoNet.load_from_checkpoint("epoch=9-step=223900.ckpt", in_channels=3, K=2)



checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
trainer = L.Trainer(
    max_epochs=100, accumulate_grad_batches=1, callbacks=[checkpoint_callback]
)


# Train
trainer.fit(
    model=model,
    train_dataloaders=DataLoader(train, batch_size=1, num_workers=23, shuffle=True),
    val_dataloaders=DataLoader(val, batch_size=1, num_workers=23),
)
# 保存参数
torch.save(model.state_dict(), "ShuffleAttention.ckpt")