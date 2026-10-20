
import pathlib
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from monai.networks.nets import AHNet


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "DATA_AVC_NPY"
ETUDES = ("Etude_1", "Etude_2", "Etude_4")

N_SLICES = 5
BATCH_SIZE = 8
NUM_WORKERS = 0

MAX_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5

# AH-Net is initialized without pretrained weights.
USE_PRETRAINED = False

CHECKPOINT_DIR = "checkpoints_ahnet_25d"


# ============================================================
# DATASET 2.5D
# ============================================================

class AVCDataset(Dataset):
    """
    Entrée :
        n_slices coupes voisines, shape (n_slices, H, W).

    Cible :
        masque de la coupe centrale, shape (1, H, W).

    Volumes attendus :
        (D, H, W).
    """

    def __init__(self, paires, augment=False, n_slices=5):
        self.paires = paires
        self.augment = augment
        self.n_slices = n_slices

        assert n_slices % 2 == 1, "n_slices doit être impair."

        self.samples = [
            (i, z)
            for i, (_, f_img, _) in enumerate(paires)
            for z in range(np.load(f_img, mmap_mode="r").shape[0])
        ]

        # Cache LRU pour limiter la consommation mémoire.
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

        if (
            img.ndim != 3
            or msk.ndim != 3
            or img.shape != msk.shape
        ):
            raise ValueError(
                "Image et masque doivent avoir la même forme 3D : "
                f"{f_img}: {img.shape}, {f_msk}: {msk.shape}"
            )

        # Normalisation Z-score par volume.
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Segmentation binaire.
        msk = (msk > 0).astype(np.float32)

        self.cache[i] = (img, msk)

        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)

        return img, msk

    def __getitem__(self, k):
        volume_idx, z = self.samples[k]
        img, msk = self._load_volume(volume_idx)

        radius = self.n_slices // 2

        # Réplique les coupes aux extrémités du volume.
        indices = [
            min(max(z + offset, 0), img.shape[0] - 1)
            for offset in range(-radius, radius + 1)
        ]

        x = torch.from_numpy(img[indices].copy()).float()
        y = torch.from_numpy(msk[z].copy()).float().unsqueeze(0)

        # Augmentation d'intensité uniquement.
        if self.augment:
            x = x * random.uniform(0.9, 1.1)
            x = x + random.uniform(-0.1, 0.1)

        return x, y


# ============================================================
# DATA MODULE
# ============================================================

class AVCDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_size=0.7,
        val_size=0.15,
        seed=42,
        etudes=ETUDES,
        n_slices=N_SLICES,
        num_workers=NUM_WORKERS,
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
                try:
                    animal_id = int(f.name[1:4])
                except ValueError as exc:
                    raise ValueError(
                        f"Impossible d'extraire l'ID animal de {f.name}. "
                        "Adaptez int(f.name[1:4]) à vos noms de fichiers."
                    ) from exc

                paires.append(((etude, animal_id), f, m))

        if not paires:
            raise RuntimeError(
                f"Aucune paire *_img.npy / *_mask.npy trouvée dans "
                f"{self.data_dir.resolve()} pour {self.etudes}."
            )

        # Split par animal, jamais par coupe, pour éviter les fuites
        # entre les ensembles d'entraînement, validation et test.
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

            print(f"{etude} : {len(animaux)} animaux")

        p_train = [p for p in paires if p[0] in g_train]
        p_val = [p for p in paires if p[0] in g_val]
        p_test = [p for p in paires if p[0] in g_test]

        if not p_train or not p_val or not p_test:
            raise ValueError(
                "Un split est vide. Il faut suffisamment d'animaux "
                "par étude pour train/validation/test. "
                f"Train={len(p_train)}, val={len(p_val)}, "
                f"test={len(p_test)}."
            )

        self.data_train = AVCDataset(
            p_train, augment=True, n_slices=self.n_slices
        )
        self.data_val = AVCDataset(
            p_val, augment=False, n_slices=self.n_slices
        )
        self.data_test = AVCDataset(
            p_test, augment=False, n_slices=self.n_slices
        )

        self.test_paths = [p[1] for p in p_test]

        print(
            "Volumes - Train:", len(p_train),
            "Val:", len(p_val),
            "Test:", len(p_test),
        )
        print(
            "Coupes - Train:", len(self.data_train),
            "Val:", len(self.data_val),
            "Test:", len(self.data_test),
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

    seuil=None :
        probabilités continues, pour la loss.

    seuil=0.5 :
        prédiction binaire, pour la métrique.
    """

    p = torch.sigmoid(logits)

    if seuil is not None:
        p = (p > seuil).float()

    y = y.float()

    inter = (p * y).sum(dim=(1, 2, 3))
    denom = (
        p.sum(dim=(1, 2, 3))
        + y.sum(dim=(1, 2, 3))
    )

    return (2 * inter + eps) / (denom + eps)


# ============================================================
# MODELE 2.5D : AH-NET
# ============================================================

class AVCSegModel(pl.LightningModule):
    """
    AH-Net en mode 2D.

    Entrée :
        (B, N_SLICES, H, W)

    Sortie :
        (B, 1, H, W)

    Les cinq coupes MRI sont utilisées comme canaux d'entrée.
    La cible reste le masque de la coupe centrale.
    """

    def __init__(
        self,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        n_slices=N_SLICES,
        dice_weight=0.7,
        encoder_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # encoder_weights est conservé pour compatibilité avec
        # l'ancien bloc d'entraînement. AH-Net utilise ici
        # pretrained=False.
        self.model = AHNet(
            spatial_dims=2,
            in_channels=n_slices,
            out_channels=1,
            psp_block_num=1,
            upsample_mode="transpose",
            pretrained=False,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        """
        AH-Net impose des contraintes sur les dimensions spatiales.
        On ajoute un padding à droite et en bas, puis on recadre
        la prédiction à la taille originale.
        """

        _, _, h, w = x.shape

        # Avec psp_block_num=1 et transpose, les dimensions doivent
        # respecter les contraintes de taille de AH-Net.
        # On utilise des dimensions >= 64, multiples de 32.
        target_h = max(64, ((h + 31) // 32) * 32)
        target_w = max(64, ((w + 31) // 32) * 32)

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )

        logits = self.model(x)

        # Retour à la taille exacte du masque original.
        logits = logits[..., :h, :w]

        return logits

    def compute_loss(self, logits, y):
        bce = self.criterion(logits, y)
        dice_loss = 1.0 - dice_par_coupe(logits, y).mean()

        return (
            self.hparams.dice_weight * dice_loss
            + (1.0 - self.hparams.dice_weight) * bce
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.compute_loss(logits, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.compute_loss(logits, y)
        dice = dice_par_coupe(
            logits, y, seuil=0.5
        ).mean()

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.compute_loss(logits, y)
        dice = dice_par_coupe(
            logits, y, seuil=0.5
        ).mean()

        self.log(
            "test_loss",
            loss,
            batch_size=x.size(0),
        )
        self.log(
            "test_dice",
            dice,
            batch_size=x.size(0),
        )

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
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        etudes=ETUDES,
        n_slices=N_SLICES,
        num_workers=NUM_WORKERS,
    )

    data_module.setup()

    # Vérification des dimensions avant l'entraînement.
    X, y = next(iter(data_module.train_dataloader()))

    print("Batch d'entrée :", X.shape)
    print("Batch cible    :", y.shape)

    # Attendu :
    # X : (B, 5, H, W)
    # y : (B, 1, H, W)

    encoder_weights = None if not USE_PRETRAINED else "imagenet"

    model = AVCSegModel(
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        n_slices=N_SLICES,
        dice_weight=0.7,
        encoder_weights=encoder_weights,
    )

    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-epoch{epoch:02d}-dice{val_dice:.4f}",
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="mps",
        devices="auto",
        callbacks=[checkpoint, early_stopping],
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    # Évaluation sur le jeu de test avec le meilleur checkpoint.
    results = trainer.test(
        model,
        datamodule=data_module,
        ckpt_path="best",
    )

    print("\nFINAL RESULTS:")
    print(results)
