import pathlib, random
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.networks.nets import DynUNet


class AVCDataset(Dataset):
    """Charge un volume (20,256,256) et son masque."""
    def __init__(self, paires, augment=False):
        self.paires = paires          # liste de (animal, chemin_img, chemin_mask)
        self.augment = augment

    def __len__(self):
        return len(self.paires)

    def __getitem__(self, k):
        _, f_img, f_msk = self.paires[k]
        img = torch.from_numpy(np.load(f_img)).float()
        msk = torch.from_numpy(np.load(f_msk)).float()
        img = (img - img.mean()) / (img.std() + 1e-8)            # z-score par volume
        if self.augment:                                         # même transformation sur image et masque
            
            img = img * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return img[None], msk[None]                                          # (1, 20, 256, 256)


class AVCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./DATA_AVC_NPY", batch_size=2, train_size=0.7, val_size=0.15,
                 seed=42, etudes=None):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.etudes = etudes                      # ex. ("Etude_1", "Etude_2", "Etude_3"), None = toutes

    def setup(self, stage=None):
        paires = []
        for f in sorted(self.data_dir.rglob("*_img.npy")):
            m = f.with_name(f.name.replace("_img.npy", "_mask.npy"))
            etude = f.parents[1].name
            if m.exists() and (self.etudes is None or etude in self.etudes):
                paires.append(((etude, int(f.name[1:4])), f, m))

        g_train, g_val, g_test = set(), set(), set()
        for etude in sorted({p[0][0] for p in paires}):          # découpage PAR ANIMAL, dans chaque étude
            animaux = sorted({p[0] for p in paires if p[0][0] == etude})
            random.Random(self.seed).shuffle(animaux)
            n_train = int(self.train_size * len(animaux))
            n_val = int(self.val_size * len(animaux))
            g_train |= set(animaux[:n_train])
            g_val |= set(animaux[n_train:n_train + n_val])
            g_test |= set(animaux[n_train + n_val:])
            print(etude, "-", len(animaux), "animaux")

        self.data_train = AVCDataset([p for p in paires if p[0] in g_train], augment=True)
        self.data_val = AVCDataset([p for p in paires if p[0] in g_val])
        self.data_test = AVCDataset([p for p in paires if p[0] in g_test])
        self.test_paths = [p[1] for p in self.data_test.paires]
        print("Volumes  - Train:", len(self.data_train), "Val:", len(self.data_val), "Test:", len(self.data_test))


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)


def dice_par_volume(logits: Tensor, y: Tensor, seuil=None, eps=1e-6) -> Tensor:
    p = torch.sigmoid(logits)
    if seuil is not None:
        p = (p > seuil).float()
    inter = (p * y).sum((1, 2, 3, 4))
    return (2 * inter + eps) / (p.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4)) + eps)


class AVCSegModel(pl.LightningModule):

    def __init__(self, lr: float = 0.001, weight_decay: float = 1e-5, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]   # axe des coupes non réduit
        self.model = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            kernel_size=[(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=strides, upsample_kernel_size=strides[1:],
            filters=[16, 32, 64, 128, 256], norm_name="instance",
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)                                     # logits (B, 1, 20, 256, 256)

    def compute_loss(self, logits, y):
        return self.criterion(logits, y) + (1 - dice_par_volume(logits, y)).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.compute_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.compute_loss(logits, y), prog_bar=True)
        self.log("val_dice", dice_par_volume(logits, y, seuil=0.5).mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_loss", self.compute_loss(logits, y))
        self.log("test_dice", dice_par_volume(logits, y, seuil=0.5).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


if __name__ == "__main__":

    pl.seed_everything(0)

    data_module = AVCDataModule(data_dir="./DATA_AVC_NPY", batch_size=2,
                            etudes=("Etude_1", "Etude_2", "Etude_3"))
    
    data_module.setup()

    X, y = next(iter(data_module.train_dataloader()))
    print("batch :", X.shape, y.shape)                          # (2, 1, 20, 256, 256)

    model = AVCSegModel()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    ckpt = ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=1)
    trainer = pl.Trainer(max_epochs=100, accelerator="auto", devices=1, callbacks=[ckpt])
    trainer.fit(model, train_dataloader, val_dataloader)

    results = trainer.test(model, data_module.test_dataloader(), ckpt_path="best")

    print("\nFINAL RESULTS:")
    print(results)