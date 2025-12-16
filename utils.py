from pathlib import Path
import shutil
import torch
import numpy as np
import tempfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Pad, Resize, ToTensor
from skimage.transform import radon, rescale
from scipy.ndimage import rotate


MNIST_TRAIN_GDRIVE = "https://drive.google.com/file/d/15G2FsYGRSpEkr5MTVofSFhKaMMIeiFhk/view?usp=drive_link"
MNIST_TEST_GDRIVE  = "https://drive.google.com/file/d/1PK1DeFpw2OomuHDoA8ZtTPWLkPHOT6u6/view?usp=drive_link"

def _gdrive_download(url_or_id: str, out_path: Path) -> None:
    import gdown

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent), suffix=".tmp") as tf:
        tmp_path = Path(tf.name)

    try:
        ok = gdown.download(url_or_id, str(tmp_path), quiet=False, fuzzy=True)
        if not ok or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"[Google Drive] Failed to download: {url_or_id}")

        shutil.move(str(tmp_path), str(out_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _ensure_mnist_amat_files(mnist_dir: str = "mnist") -> None:
    mnist_dir = Path(mnist_dir)
    train_path = mnist_dir / "mnist_train.amat"
    test_path = mnist_dir / "mnist_test.amat"

    if not train_path.is_file():
        print(f"[MNIST] '{train_path}' not found. Downloading MNIST train file from Google Drive...")
        _gdrive_download(MNIST_TRAIN_GDRIVE, train_path)
        print(f"[MNIST] Download complete: {train_path}")

    if not test_path.is_file():
        print(f"[MNIST] '{test_path}' not found. Downloading MNIST test file from Google Drive...")
        _gdrive_download(MNIST_TEST_GDRIVE, test_path)
        print(f"[MNIST] Download complete: {test_path}")


class MnistDataset(Dataset):
    def __init__(self, mode, num_train_data=10000, seed=1):
        assert num_train_data + 12000 <= 45000
        assert mode in ['train', 'validation', 'test']
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        _ensure_mnist_amat_files("mnist")

        if mode == "test":
            file = "mnist/mnist_test.amat"
        else:
            file = "mnist/mnist_train.amat"

        data = np.loadtxt(file)
        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)

        # Shuffle the images
        indices = np.arange(num_train_data + 12000)
        np.random.shuffle(indices)
        if mode == 'train':
            images = images[indices[:num_train_data]]
            data = data[indices[:num_train_data]]
        elif mode == 'validation':
            images = images[indices[num_train_data:]]
            data = data[indices[num_train_data:]]

        if mode == 'test' or mode == 'validation':
            pad = Pad((0, 0, 1, 1), fill=0)
            resize1 = Resize(87)  # to upsample
            resize2 = Resize(29)  # to downsample
            totensor = ToTensor()

            self.images = torch.empty((images.shape[0], 1, 29, 29))
            for i in range(images.shape[0]):
                img = images[i]
                img = Image.fromarray(img, mode='F')
                r = (np.random.rand() * 360.)
                self.images[i] = totensor(resize2(resize1(pad(img)).rotate(r, Image.BILINEAR))).reshape(1, 29, 29)
        else:
            self.images = torch.zeros((images.shape[0], 1, 29, 29))
            self.images[:, :, :28, :28] = torch.tensor(images).reshape(-1, 1, 28, 28)

        self.labels = data[:, -1].astype(np.int64)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)


def cal_mean_std(ary):
    ary_np = np.array(ary)
    return np.mean(ary_np), np.std(ary_np)

def print_run_config(args, device, base_seed):
    print("\n" + "=" * 34)
    print("RA Training Configuration")
    print("-" * 34)
    print(f"Device          : {device}")
    print(f"Group           : {args.group}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Train samples   : {args.num_train_data}")
    print(f"Num seeds       : {args.n_seeds}")
    print(f"Seed range      : {base_seed} ~ {base_seed + args.n_seeds - 1}")
    print("=" * 34 + "\n")