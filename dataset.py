# %%
import torch
import cv2
import glob, os, re
import numpy as np


# %%
def read_pfm(file):
    with open(file, "rb") as f:
        # 读取文件的头部以确定格式、大小
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("latin-1"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        # 缩放系数
        scale = float(f.readline().rstrip())
        if scale < 0:  # 小端字节序
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # 大端字节序

        # 读取数据
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        # 重塑并返回
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM文件的坐标系是左下角
        data = data / scale
        return data


# %%
class flythings3d(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode="train", downsampling=2):
        if mode == "train":
            root_dir = os.path.join(root_dir, "frames_disparity/TRAIN")
        elif mode == "val":
            root_dir = os.path.join(root_dir, "frames_disparity/validation")
        elif mode == "test":
            root_dir = os.path.join(root_dir, "frames_disparity/TEST")
        self.files_list = glob.glob(f"{root_dir}/**/*.png", recursive=True)
        self.files_list = [file for file in self.files_list if "/left/" in file]
        self.downsampling = downsampling

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        left_disp_path = self.files_list[idx]
        right_disp_path = left_disp_path.replace("/left/", "/right/")
        left_img_path = left_disp_path.replace(
            "frames_disparity", "frames_cleanpass"
        ).replace(".png", ".png")
        right_img_path = left_img_path.replace("/left/", "/right/")

        left_img = cv2.imread(left_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        right_img = cv2.imread(right_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        left_disp_0 = cv2.imread(left_disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        # 读取右眼视差图

        right_disp_0 = cv2.imread(right_disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000

        left_disp = np.expand_dims(left_disp_0 / self.downsampling, axis=-1)
        right_disp = np.expand_dims(right_disp_0 / self.downsampling, axis=-1)

        # left_disp = np.expand_dims(
        #    read_pfm(left_disp_path) / self.downsampling, axis=-1
        # )
        # right_disp = np.expand_dims(
        #    read_pfm(right_disp_path) / self.downsampling, axis=-1
        # )

        sample = np.concatenate([left_img, right_img, left_disp, right_disp], axis=-1)
        if self.downsampling != 1:
            sample = cv2.resize(
                sample,
                dsize=(
                    sample.shape[1] // self.downsampling,
                    sample.shape[0] // self.downsampling,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        sample = np.transpose(sample, (2, 0, 1))

        # conver to torch tensor
        sample = torch.from_numpy(sample)
        return sample.float()


class Crestereo(torch.utils.data.Dataset):
    def __init__(self, root_dir, downsampling=3):
        self.files_list = glob.glob(f"{root_dir}/**/*.jpg", recursive=True)
        self.files_list = [file for file in self.files_list if "_left.jpg" in file]
        self.files_list = self.files_list[50000:]
        self.downsampling = downsampling

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        left_img_path = self.files_list[idx]
        right_img_path = left_img_path.replace("_left.jpg", "_right.jpg")
        left_disp_path = left_img_path.replace("_left.jpg", "_left.disp.png")
        right_disp_path = left_img_path.replace("_left.jpg", "_right.disp.png")

        left_img = cv2.imread(left_img_path).astype(float) / 255
        right_img = cv2.imread(right_img_path).astype(float) / 255
        left_disp = cv2.imread(left_disp_path)
        left_disp = (
                np.expand_dims(
                    cv2.cvtColor(left_disp, cv2.COLOR_BGR2GRAY).astype(float), axis=-1
                )
                / self.downsampling
        )
        right_disp = cv2.imread(right_disp_path)
        right_disp = (
                np.expand_dims(
                    cv2.cvtColor(right_disp, cv2.COLOR_BGR2GRAY).astype(float), axis=-1
                )
                / self.downsampling
        )

        sample = np.concatenate([left_img, right_img, left_disp, right_disp], axis=-1)
        if self.downsampling != 1:
            sample = cv2.resize(
                sample,
                dsize=(
                    sample.shape[1] // self.downsampling,
                    sample.shape[0] // self.downsampling,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        sample = np.transpose(sample, (2, 0, 1))
        # conver to torch tensor
        sample = torch.from_numpy(sample)

        return sample.float()


# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = flythings3d("data/flythings3d/", mode="test")
    sample = data[0]

    left = sample[:3, ...]
    left = np.transpose(left, (1, 2, 0))
    right = sample[3:6, ...]
    right = np.transpose(right, (1, 2, 0))
    left_disp = sample[6, ...]
    right_disp = sample[7, ...]

    plt.imshow(left)
    plt.show()

    plt.imshow(right)
    plt.show()

    plt.imshow(left_disp)
    plt.show()

    plt.imshow(right_disp)
    plt.show()
