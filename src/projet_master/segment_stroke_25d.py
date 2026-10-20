# import pathlib, random
# import numpy as np
# import torch
# from torch import nn, Tensor
# from torch.utils.data import Dataset, DataLoader

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from monai.networks.nets import DynUNet


# class AVCDataset(Dataset):
#     """Charge un volume (20,256,256) et son masque."""
#     def __init__(self, paires, augment=False):
#         self.paires = paires          # liste de (animal, chemin_img, chemin_mask)
#         self.augment = augment

#     def __len__(self):
#         return len(self.paires)

#     def __getitem__(self, k):
#         _, f_img, f_msk = self.paires[k]
#         img = torch.from_numpy(np.load(f_img)).float()
#         msk = torch.from_numpy(np.load(f_msk)).float()
#         img = (img - img.mean()) / (img.std() + 1e-8)            # z-score par volume
#         if self.augment:                                         # même transformation sur image et masque
            
#             img = img * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
#         return img[None], msk[None]                                          # (1, 20, 256, 256)


# class AVCDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir="./FULLMRI", batch_size=2, train_size=0.7, val_size=0.15,
#                  seed=42, etudes=None):
#         super().__init__()
#         self.data_dir = pathlib.Path(data_dir)
#         self.batch_size = batch_size
#         self.train_size = train_size
#         self.val_size = val_size
#         self.seed = seed
#         self.etudes = etudes                      # ex. ("Etude_1", "Etude_2", "Etude_3"), None = toutes

#     def setup(self, stage=None):
#         paires = []
#         for f in sorted(self.data_dir.rglob("*_img.npy")):
#             m = f.with_name(f.name.replace("_img.npy", "_mask.npy"))
#             etude = f.parents[1].name
#             if m.exists() and (self.etudes is None or etude in self.etudes):
#                 paires.append(((etude, int(f.name[1:4])), f, m))

#         g_train, g_val, g_test = set(), set(), set()
#         for etude in sorted({p[0][0] for p in paires}):          # découpage PAR ANIMAL, dans chaque étude
#             animaux = sorted({p[0] for p in paires if p[0][0] == etude})
#             random.Random(self.seed).shuffle(animaux)
#             n_train = int(self.train_size * len(animaux))
#             n_val = int(self.val_size * len(animaux))
#             g_train |= set(animaux[:n_train])
#             g_val |= set(animaux[n_train:n_train + n_val])
#             g_test |= set(animaux[n_train + n_val:])
#             print(etude, "-", len(animaux), "animaux")

#         self.data_train = AVCDataset([p for p in paires if p[0] in g_train], augment=True)
#         self.data_val = AVCDataset([p for p in paires if p[0] in g_val])
#         self.data_test = AVCDataset([p for p in paires if p[0] in g_test])
#         self.test_paths = [p[1] for p in self.data_test.paires]
#         print("Volumes  - Train:", len(self.data_train), "Val:", len(self.data_val), "Test:", len(self.data_test))


#     def train_dataloader(self):
#         return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)


# def dice_par_volume(logits: Tensor, y: Tensor, seuil=None, eps=1e-6) -> Tensor:
#     p = torch.sigmoid(logits)
#     if seuil is not None:
#         p = (p > seuil).float()
#     inter = (p * y).sum((1, 2, 3, 4))
#     return (2 * inter + eps) / (p.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4)) + eps)


# class AVCSegModel(pl.LightningModule):

#     def __init__(self, lr: float = 0.001, weight_decay: float = 1e-5, *args, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()

#         strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]   # axe des coupes non réduit
#         self.model = DynUNet(
#             spatial_dims=3, in_channels=1, out_channels=1,
#             kernel_size=[(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
#             strides=strides, upsample_kernel_size=strides[1:],
#             filters=[16, 32, 64, 128, 256], norm_name="instance",
#         )
#         self.criterion = nn.BCEWithLogitsLoss()

#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)                                     # logits (B, 1, 20, 256, 256)

#     def compute_loss(self, logits, y):
#         return self.criterion(logits, y) + (1 - dice_par_volume(logits, y)).mean()

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         loss = self.compute_loss(self(x), y)
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         self.log("val_loss", self.compute_loss(logits, y), prog_bar=True)
#         self.log("val_dice", dice_par_volume(logits, y, seuil=0.5).mean(), prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         self.log("test_loss", self.compute_loss(logits, y))
#         self.log("test_dice", dice_par_volume(logits, y, seuil=0.5).mean())

#     def configure_optimizers(self):
#         return torch.optim.Adam(
#             self.parameters(),
#             lr=self.hparams.lr,
#             weight_decay=self.hparams.weight_decay
#         )


# if __name__ == "__main__":

#     pl.seed_everything(0)

#     data_module = AVCDataModule(data_dir="./FULLMRI", batch_size=2,
#                             etudes=("Etude_1", "Etude_2", "Etude_3"))
    
#     data_module.setup()

#     X, y = next(iter(data_module.train_dataloader()))
#     print("batch :", X.shape, y.shape)                          # (2, 1, 20, 256, 256)

#     model = AVCSegModel()

#     train_dataloader = data_module.train_dataloader()
#     val_dataloader = data_module.val_dataloader()

#     ckpt = ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=1)
#     trainer = pl.Trainer(max_epochs=100, accelerator="auto", devices=1, callbacks=[ckpt])
#     trainer.fit(model, train_dataloader, val_dataloader)

#     results = trainer.test(model, data_module.test_dataloader(), ckpt_path="best")

#     print("\nFINAL RESULTS:")
#     print(results)


import pathlib
import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.networks.nets import UNet


# ============================================================
# DATASET 2.5D
# ============================================================

class AVCDataset(Dataset):
    """
    Entrée : 5 coupes IRM voisines (5, 256, 256)
    Cible  : masque de la coupe centrale (1, 256, 256)
    """

    def __init__(self, paires, augment=False, n_slices=5):
        self.paires = paires
        self.augment = augment
        self.n_slices = n_slices

        assert n_slices % 2 == 1, "n_slices doit être impair."

        # Chaque volume fournit plusieurs exemples 2.5D.
        self.samples = [
            (i, z)
            for i, (_, f_img, _) in enumerate(paires)
            for z in range(np.load(f_img, mmap_mode="r").shape[0])
        ]

        # Cache LRU limité pour éviter de relire les volumes
        # à chaque coupe, sans conserver tous les volumes en RAM.
        self.cache = OrderedDict()
        self.max_cache = 2

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]

        _, f_img, f_msk = self.paires[i]

        img = np.load(f_img).astype(np.float32)
        msk = np.load(f_msk).astype(np.float32)

        img = (img - img.mean()) / (img.std() + 1e-8)

        self.cache[i] = (img, msk)

        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)

        return img, msk

    def __getitem__(self, k):
        volume_idx, z = self.samples[k]
        img, msk = self._load_volume(volume_idx)

        radius = self.n_slices // 2
        indices = [
            min(max(z + offset, 0), img.shape[0] - 1)
            for offset in range(-radius, radius + 1)
        ]

        # (n_slices, H, W)
        x = torch.from_numpy(img[indices].copy())

        # Masque de la coupe centrale : (1, H, W)
        y = torch.from_numpy(msk[z].copy()).float().unsqueeze(0)

        if self.augment:
            # Transformations d'intensité uniquement.
            x = x * random.uniform(0.9, 1.1)
            x = x + random.uniform(-0.1, 0.1)

        return x, y


# ============================================================
# DATA MODULE
# ============================================================

class AVCDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir="DATA_AVC_NPY",
        batch_size=8,
        train_size=0.7,
        val_size=0.15,
        seed=42,
        etudes=None,
        n_slices=5,
        num_workers=0,
    ):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.etudes = etudes
        self.n_slices = n_slices
        self.num_workers = num_workers

    def setup(self, stage=None):
        paires = []

        for f in sorted(self.data_dir.rglob("*_img.npy")):
            m = f.with_name(f.name.replace("_img.npy", "_mask.npy"))
            etude = f.parents[1].name

            if m.exists() and (
                self.etudes is None or etude in self.etudes
            ):
                # Identifiant animal : à adapter si le nom
                # des fichiers utilise une autre convention.
                animal_id = int(f.name[1:4])
                paires.append(((etude, animal_id), f, m))

        g_train, g_val, g_test = set(), set(), set()

        for etude in sorted({p[0][0] for p in paires}):
            animaux = sorted({
                p[0] for p in paires if p[0][0] == etude
            })

            rng = random.Random(self.seed)
            rng.shuffle(animaux)

            n_train = int(self.train_size * len(animaux))
            n_val = int(self.val_size * len(animaux))

            g_train.update(animaux[:n_train])
            g_val.update(animaux[n_train:n_train + n_val])
            g_test.update(animaux[n_train + n_val:])

            print(f"{etude} - {len(animaux)} animaux")

        p_train = [p for p in paires if p[0] in g_train]
        p_val = [p for p in paires if p[0] in g_val]
        p_test = [p for p in paires if p[0] in g_test]

        self.data_train = AVCDataset(
            p_train, augment=True, n_slices=self.n_slices
        )
        self.data_val = AVCDataset(
            p_val, n_slices=self.n_slices
        )
        self.data_test = AVCDataset(
            p_test, n_slices=self.n_slices
        )

        self.test_paths = [p[1] for p in p_test]

        print(
            "Volumes - Train:", len(p_train),
            "Val:", len(p_val),
            "Test:", len(p_test)
        )
        print(
            "Coupes - Train:", len(self.data_train),
            "Val:", len(self.data_val),
            "Test:", len(self.data_test)
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# ============================================================
# DICE 2D PAR COUPE
# ============================================================

def dice_par_coupe(
    logits: Tensor,
    y: Tensor,
    seuil=None,
    eps=1e-6,
):
    """
    logits, y : (B, 1, H, W)
    Retourne un Dice par coupe : (B,)
    """

    p = torch.sigmoid(logits)

    if seuil is not None:
        p = (p > seuil).float()

    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))

    dice = (2 * inter + eps) / (denom + eps)

    # Convention : Dice = 1 si la prédiction et la cible
    # sont toutes deux vides.
    return dice


# ============================================================
# MODELE U-NET 2.5D
# ============================================================

class AVCSegModel(pl.LightningModule):

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-5,
        n_slices=5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # U-Net 2D : les coupes voisines sont des canaux.
        self.model = UNet(
            spatial_dims=2,
            in_channels=n_slices,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        # Entrée : (B, n_slices, H, W)
        # Sortie : (B, 1, H, W)
        return self.model(x)

    def compute_loss(self, logits, y):
        bce = self.criterion(logits, y)
        dice_loss = 1 - dice_par_coupe(logits, y).mean()
        return bce + dice_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        self.log(
            "train_loss", loss,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.compute_loss(logits, y)
        dice = dice_par_coupe(logits, y, seuil=0.5).mean()

        self.log(
            "val_loss", loss,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log(
            "val_dice", dice,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.compute_loss(logits, y)
        dice = dice_par_coupe(logits, y, seuil=0.5).mean()

        self.log("test_loss", loss, batch_size=x.size(0))
        self.log("test_dice", dice, batch_size=x.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


# ============================================================
# ENTRAINEMENT
# ============================================================

if __name__ == "__main__":

    pl.seed_everything(42, workers=True)

    data_module = AVCDataModule(
        data_dir="DATA_AVC_NPY",
        batch_size=8,
        etudes=("Etude_1", "Etude_2", "Etude_4"),
        n_slices=5,
        num_workers=0,
    )

    data_module.setup()

    X, y = next(iter(data_module.train_dataloader()))
    print("Batch :", X.shape, y.shape)
    # X : (8, 5, 256, 256)
    # y : (8, 1, 256, 256)

    model = AVCSegModel(
        lr=1e-3,
        weight_decay=1e-5,
        n_slices=5,
    )

    ckpt = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-epoch{epoch:02d}-dice{val_dice:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt],
    )

    trainer.fit(model, data_module)

    results = trainer.test(
        model,
        datamodule=data_module,
        ckpt_path="best",
    )

    print("\nFINAL RESULTS:")
    print(results)
