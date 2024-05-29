# %%
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch import loggers as pl_loggers
import os
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SKAttention import SKAttention

# %%
def plot_figure(
    left: torch.Tensor,
    right: torch.Tensor,
    disp_gt: torch.Tensor,
    disp_pred: torch.Tensor,
) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close("all")
    fig, ax = plt.subplots(ncols=2, nrows=2, dpi=300)
    left = (torch.moveaxis(left, 0, 2) + 1) / 2
    right = (torch.moveaxis(right, 0, 2) + 1) / 2
    disp_gt = torch.moveaxis(disp_gt, 0, 2)
    disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left)
    ax[0, 1].imshow(right)
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max())
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max())
    ax[0, 0].title.set_text("Left")
    ax[0, 1].title.set_text("Right")
    ax[1, 0].title.set_text("Ground truth disparity")
    ax[1, 1].title.set_text("Predicted disparity")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def VolumeCompute(
    ref: torch.Tensor, sft: torch.Tensor, D: int = 32, side: str = "left"
) -> torch.Tensor:
    batch, channel, height, width = ref.size()
    cost = torch.zeros(batch, channel, D, height, width)
    cost = cost.type_as(ref)

    cost[:, :, 0, :, :] = ref - sft
    for idx in range(1, D):
        if side == "left":
            cost[:, :, idx, :, idx:] = ref[:, :, :, idx:] - sft[:, :, :, :-idx]
        if side == "right":
            cost[:, :, idx, :, :-idx] = ref[:, :, :, :-idx] - sft[:, :, :, idx:]

    cost = cost.contiguous()
    return cost


def soft_argmin(cost: torch.Tensor, max_downsampled_disps: int) -> torch.Tensor:
    """
    Soft argmin function described in the original paper.  The disparity grid creates the first 'd' value in equation 2 while
    cost is the C_i(d) term.  The exp/sum(exp) == softmax function.
    """

    disparity_softmax = F.softmax(-cost, dim=1)
    # TODO: Bilinear interpolate the disparity dimension back to D to perform the proper d*exp(-C_i(d))

    disparity_grid = torch.linspace(
        0, max_downsampled_disps, disparity_softmax.size(1)
    ).reshape(1, -1, 1, 1)
    disparity_grid = disparity_grid.type_as(disparity_softmax)

    disp = torch.sum(disparity_softmax * disparity_grid, dim=1, keepdim=True)

    return disp


def robust_loss(
    x: torch.Tensor, alpha: float, c: float
) -> torch.Tensor:  # pylint: disable=invalid-name
    """
    A General and Adaptive Robust Loss Function (https://arxiv.org/abs/1701.03077)
    """

    f: torch.Tensor = (abs(alpha - 2) / alpha) * (
        torch.pow(torch.pow(x / c, 2) / abs(alpha - 2) + 1, alpha / 2) - 1
    )  # pylint: disable=invalid-name
    return f


class ResBlock(nn.Module):
    """
    Just a note, in the original paper, there is no discussion about padding; however, both the ZhiXuanLi and the X-StereoLab implementation using padding.
    This does make sense to maintain the image size after the feature extraction has occured.

    X-StereoLab uses a simple Res unit with a single conv and summation while ZhiXuanLi uses the original residual unit implementation.
    This class also uses the original implementation with 2 layers of convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """
        Original Residual Unit: https://arxiv.org/pdf/1603.05027.pdf (Fig 1. Left)
        """

        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activation_1(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)

        # I'm not really sure why the type definition is required here... nn.Conv2d already returns type Tensor...
        # So res should be of type torch.Tensor AND x is already defined as type torch.Tensor.
        out: torch.Tensor = res + x
        out = self.activation_2(out)

        return out

class CrossAttentionWithSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super(CrossAttentionWithSE, self).__init__()

        # Define your cross attention module here

        # self.se = SKAttention(channel=channels, reduction=reduction)
        self.se = SEAttention(channel=channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply cross attention
        # cross_attention_output = your_cross_attention_module(x)

        # Apply SEAttention
        se_output = self.se(x)

        # Combine the outputs (you may want to adjust this based on your specific architecture)
        output = se_output

        return output

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int) -> None:
        super().__init__()

        self.K = K

        net = []

        # K downsampling nets
        for _ in range(self.K):
            net.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                )
            )
            in_channels = out_channels

        # 6 residual blocks
        for _ in range(6):
            net.append(
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )

            )
        net.append(
             CrossAttentionWithSE(out_channels)
                      )

        # final output net
        net.append(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostVolume(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, D) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.D = D

        net = []

        for _ in range(4):
            net.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            net.append(nn.BatchNorm3d(num_features=out_channels, eps=1))
            net.append(nn.LeakyReLU(negative_slope=0.2))

            in_channels = out_channels

        net.append(
            nn.Conv3d(
                in_channels=out_channels, out_channels=1, kernel_size=3, padding=1
            )
        )

        self.net = nn.Sequential(*net)

    def forward(
        self, x: [torch.Tensor, torch.Tensor], side: str = "left"
    ) -> torch.Tensor:
        left, right = x
        cost = VolumeCompute(left, right, side=side)
        cost = self.net(cost)
        cost = torch.squeeze(cost, dim=1)

        return cost


class Refinement(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        dilations = [1, 2, 3, 8, 1, 1]
        net = []

        net.append(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
            )
        )

        for idx, dilation in enumerate(dilations):
            net.append(
                ResBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
            )

        net.append(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class StereoNet(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        K: int = 3,
        D: int = 256,
        refinement_layers: int = 3,
        feature_extractor_filters: int = 32,
        cost_volumizer_filters: int = 32,
        mask: bool = True,
    ):
        super().__init__()

        self.refinement_layers = refinement_layers
        self.in_channels = in_channels
        self.feature_extractor_filters = feature_extractor_filters
        self.cost_volumizer_filters = cost_volumizer_filters
        self.D = D
        self.mask = mask
        self.count = 0
        self.mean_loss = 0.0


        # D = (D + 1) // (2**K)

        self.featureExtractor = FeatureExtractor(
            in_channels=in_channels, out_channels=feature_extractor_filters, K=3
        )

        self.costVolume = CostVolume(
            in_channels=feature_extractor_filters,
            out_channels=cost_volumizer_filters,
            D=D,
        )

        self.refiners = nn.ModuleList()
        for _ in range(self.refinement_layers):
            self.refiners.append(Refinement(in_channels=in_channels + 1))

    def forward_pyramid(self, batch: torch.Tensor, side: str = "left"):
        if side == "left":
            ref = batch[:, : self.in_channels, ...]
            sft = batch[:, self.in_channels : self.in_channels * 2, ...]
        elif side == "right":
            ref = batch[:, self.in_channels : self.in_channels * 2, ...]
            sft = batch[:, : self.in_channels, ...]

        ref_feature = self.featureExtractor(ref)
        sft_feature = self.featureExtractor(sft)

        cost = self.costVolume((ref_feature, sft_feature), side=side)
        disparity_pyramid = [soft_argmin(cost, self.D)]

        for idx, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.refinement_layers) / (2**idx)
            new_h, new_w = int(ref.size()[2] // scale), int(ref.size()[3] // scale)
            ref_rescaled = F.interpolate(
                ref, [new_h, new_w], mode="bilinear", align_corners=True
            )
            disparity_low_rescaled = F.interpolate(
                disparity_pyramid[-1],
                [new_h, new_w],
                mode="bilinear",
                align_corners=True,
            )
            refined_disparity = F.relu(
                refiner(torch.cat((ref_rescaled, disparity_low_rescaled), dim=1))
                + disparity_low_rescaled
            )
            disparity_pyramid.append(refined_disparity)
        return disparity_pyramid

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        disparities = self.forward_pyramid(batch, side="left")
        return disparities[-1]

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        height, width = batch.size()[-2:]

        # Non-uniform because the sizes of each of the list entries returned from the forward_pyramid aren't the same
        disp_pred_left_nonuniform = self.forward_pyramid(batch, side="left")
        disp_pred_right_nonuniform = self.forward_pyramid(batch, side="right")

        for idx, (disparity_left, disparity_right) in enumerate(
            zip(disp_pred_left_nonuniform, disp_pred_right_nonuniform)
        ):
            disp_pred_left_nonuniform[idx] = F.interpolate(
                disparity_left, [height, width], mode="bilinear", align_corners=True
            )
            disp_pred_right_nonuniform[idx] = F.interpolate(
                disparity_right, [height, width], mode="bilinear", align_corners=True
            )

        disp_pred_left = torch.stack(disp_pred_left_nonuniform, dim=0)
        disp_pred_right = torch.stack(disp_pred_right_nonuniform, dim=0)

        def _tiler(tensor: torch.Tensor, matching_size: [] = None) -> torch.Tensor:
            if matching_size is None:
                matching_size = [disp_pred_left.size()[0], 1, 1, 1, 1]
            return tensor.tile(matching_size)

        disp_gt_left = _tiler(batch[:, -2, ...])
        # print(disp_gt_left)
        disp_gt_right = _tiler(batch[:, -1, ...])

        if self.mask:
            left_mask = (disp_gt_left < self.D).detach()
            right_mask = (disp_gt_right < self.D).detach()

            loss_left = torch.mean(
                robust_loss(
                    disp_gt_left[left_mask] - disp_pred_left[left_mask], alpha=1, c=2
                )
            )
            loss_right = torch.mean(
                robust_loss(
                    disp_gt_right[right_mask] - disp_pred_right[right_mask],
                    alpha=1,
                    c=2,
                )
            )
        else:
            loss_left = torch.mean(
                robust_loss(disp_gt_left - disp_pred_left, alpha=1, c=2)
            )
            loss_right = torch.mean(
                robust_loss(disp_gt_right - disp_pred_right, alpha=1, c=2)
            )

        loss = (loss_left + loss_right) / 2

        self.log(
            "train_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_epoch",
            F.l1_loss(disp_pred_left[-1], disp_gt_left[-1]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        disp_pred = self(batch[:, : self.in_channels * 2, ...])
        disp_gt = batch[:, self.in_channels * 2 : self.in_channels * 2 + 1, ...]

        loss = F.l1_loss(disp_pred, disp_gt)
        self.log("val_loss_epoch", loss, on_epoch=True, logger=True)
        if batch_idx == 0:
            fig = plot_figure(
                batch[0, : self.in_channels, ...].detach().cpu(),
                batch[0, self.in_channels : self.in_channels * 2, ...].detach().cpu(),
                batch[0, -2:-1, ...].detach().cpu(),
                disp_pred[0].detach().cpu(),
            )
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.moveaxis(data, 2, 0)

            tensorboard: pl_loggers.TensorBoardLogger = self.logger
            tensorboard.experiment.add_image(
                "generated_images", data, self.current_epoch
            )
            save_folder = "pic/V13-ShuffleAttention"
            os.makedirs(save_folder, exist_ok=True)
            file_name = os.path.join(save_folder, f"generated_image_epoch_{self.current_epoch}_{loss}.png")
            fig.savefig(file_name)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.count += 1
        disp_pred = self(batch[:, : self.in_channels * 2, ...])
        disp_gt = batch[:, self.in_channels * 2 : self.in_channels * 2 + 1, ...]
        loss = F.mse_loss(disp_pred, disp_gt)
        self.mean_loss += loss.item()
        # print("Batch count:", self.count)
        # print("Mean loss:", self.mean_loss)
        # return self.mean_loss / self.count

    def get_meanloss(self):
        return self.mean_loss / self.count


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.9
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "ExponentialDecayLR",
            },
        }
        return config


# %%
if __name__ == "__main__":
    from dataset import *

    train = flythings3d("data/flythings3d/", mode="train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StereoNet(in_channels=3, K=3, D=256).to(device)
    batch = train[1].unsqueeze(0).to(device)
    model.forward_pyramid(batch)