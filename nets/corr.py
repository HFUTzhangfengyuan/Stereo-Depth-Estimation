import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F
from .utils.utils import bilinear_sampler, coords_grid


class AGCL:
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """

    def __init__(self, fmap1, fmap2, att=None):
        # 初始化AGCL层的对象
        # 参数：
        #     - fmap1: 第一个特征图
        #     - fmap2: 第二个特征图
        #     - att: 高级注意力模块（可选）
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.att = att

        # 生成用于计算相关性的坐标矩阵
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3]).to(
            fmap1.device
        )

    def __call__(self, flow, extra_offset, small_patch=False, iter_mode=False):
        """
        调用函数时执行的操作。

        参数：
        flow: 流
        extra_offset: 额外偏移量
        small_patch: 是否为小补丁，默认为False
        iter_mode: 是否为迭代模式，默认为False

        返回：
        corr: 相关系数
        """
        if iter_mode:
            corr = self.corr_iter(self.fmap1, self.fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(
                self.fmap1, self.fmap2, flow, extra_offset, small_patch
            )
        return corr

    def get_correlationget_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
        """
        计算两个特征之间的相关性。

        参数：
            left_feature (Tensor): 左特征张量，形状为(N, C, H, W)。
            right_feature (Tensor): 右特征张量，形状为(N, C, H, W)。
            psize (tuple): 均值池化窗口的大小，默认为(3, 3)。
            dilate (tuple): 拓展大小，默认为(1, 1)。

        返回：
            corr_final (Tensor): 相关性张量，形状为(1, -1, H, W)。
        """

        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = F.pad(right_feature, pad_width=(
            (0, 0), (0, 0), (pady, pady), (padx, padx)), mode="replicate")

        right_slid = F.sliding_window(
            right_pad, kernel_size=(H, W), stride=(di_y, di_x))
        right_slid = right_slid.reshape(N, C, -1, H, W)
        right_slid = F.transpose(right_slid, (0, 2, 1, 3, 4))
        right_slid = right_slid.reshape(-1, C, H, W)

        corr_mean = F.mean(left_feature * right_slid, axis=1, keepdims=True)
        corr_final = corr_mean.reshape(1, -1, H, W)

        return corr_final

    def corr_iter(self, left_feature, right_feature, flow, small_patch):
        """
        迭代计算相关性的函数

        Args:
            left_feature (Tensor): 左特征图
            right_feature (Tensor): 右特征图
            flow (Tensor): 光流
            small_patch (bool): 是否为小图像

        Returns:
            final_corr (Tensor): 最终的相关性图
        """

        # 将坐标加上光流
        coords = self.coords + flow
        # 将坐标进行转置
        coords = F.transpose(coords, (0, 2, 3, 1))
        # 使用双线性采样来生成右特征图
        right_feature = bilinear_sampler(right_feature, coords)

        # 判断是否为小图像
        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        # 获取左特征图和右特征图的通道数、高度和宽度
        N, C, H, W = left_feature.shape
        # 将左特征图和右特征图按通道分割
        lefts = F.split(left_feature, 4, axis=1)
        rights = F.split(right_feature, 4, axis=1)

        # 计算相关性
        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(
                lefts[i], rights[i], psize_list[i], dilate_list[i]
            )
            corrs.append(corr)

        # 将计算得到的相关性进行拼接
        final_corr = F.concat(corrs, axis=1)

        return final_corr

    def corr_att_offset(
            self, left_feature, right_feature, flow, extra_offset, small_patch
    ):
        """
        函数功能：相关性偏移

        参数：
        left_feature: 左特征
        right_feature: 右特征
        flow: 光流
        extra_offset: 额外偏移
        small_patch: 是否为小图像

        返回值：
        final_corr: 最终相关性
        """

        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature = F.reshape(
                F.transpose(left_feature, (0, 2, 3, 1)), (N, H * W, C)
            )  # 'n c h w -> n (h w) c'
            right_feature = F.reshape(
                F.transpose(right_feature, (0, 2, 3, 1)), (N, H * W, C)
            )  # 'n c h w -> n (h w) c'
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = [
                F.transpose(F.reshape(x, (N, H, W, C)), (0, 3, 1, 2))
                for x in [left_feature, right_feature]
            ]

        lefts = F.split(left_feature, 4, axis=1)
        rights = F.split(right_feature, 4, axis=1)

        C = C // 4

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        search_num = 9
        extra_offset = F.transpose(
            F.reshape(extra_offset, (N, search_num, 2, H, W)), (0, 1, 3, 4, 2)
        )  # [N, search_num, 1, 1, 2]

        corrs = []
        for i in range(len(psize_list)):
            left_feature, right_feature = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex

            x_grid, y_grid = np.meshgrid(
                np.arange(-rx, rx + 1, dilatex), np.arange(-ry, ry + 1, dilatey)
            )
            y_grid, x_grid = mge.tensor(y_grid, device=self.fmap1.device), mge.tensor(
                x_grid, device=self.fmap1.device
            )
            offsets = F.transpose(
                F.reshape(F.stack((x_grid, y_grid)), (2, -1)), (1, 0)
            )  # [search_num, 2]
            offsets = F.expand_dims(offsets, (0, 2, 3))
            offsets = offsets + extra_offset

            coords = self.coords + flow  # [N, 2, H, W]
            coords = F.transpose(coords, (0, 2, 3, 1))  # [N, H, W, 2]
            coords = F.expand_dims(coords, 1) + offsets
            coords = F.reshape(coords, (N, -1, W, 2))  # [N, search_num*H, W, 2]

            right_feature = bilinear_sampler(
                right_feature, coords
            )  # [N, C, search_num*H, W]
            right_feature = F.reshape(
                right_feature, (N, C, -1, H, W)
            )  # [N, C, search_num, H, W]

            left_feature = F.expand_dims(left_feature, 2)
            corr = F.mean(left_feature * right_feature, axis=1)

            corrs.append(corr)

        final_corr = F.concat(corrs, axis=1)

        return final_corr


