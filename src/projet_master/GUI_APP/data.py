import os
import pathlib
import unicodedata

import brkraw
import numpy as np


def find_dataset_files(dir_path: str) -> list[str]:
    """Recherche les fichiers .PvDatasets dans un dossier."""
    if not dir_path or not os.path.isdir(dir_path):
        return []

    return sorted(
        os.path.join(dir_path, entry.name)
        for entry in os.scandir(dir_path)
        if entry.name.lower().endswith(".pvdatasets")
        and entry.is_file()
    )


def load_data(file_path: str, scan_id=None):
    """
    Charge les données IRM d'un scan dans un fichier PvDataset.

    Returns:
        data: tableau IRM ou None
        selected_scan: informations du scan sélectionné ou None
        all_scans: informations de tous les scans
    """
    loader = brkraw.load(file_path)
    all_scans = loader.info(scope="scan", as_dict=True)

    if not all_scans:
        return None, None, all_scans

    selected_scan_id = scan_id if scan_id in all_scans else next(iter(all_scans))
    scan_info = all_scans[selected_scan_id]
    return np.asarray(loader.get_dataobj(selected_scan_id)), scan_info, all_scans


def scan_label(scan_id, scan_info: dict) -> str:
    """Construit un libellé lisible en conservant l'identifiant du scan."""
    protocol = scan_info.get("Protocol", "Protocol inconnu")
    return f"{scan_id} - {protocol}"


def default_scan_index(all_scans: dict) -> int:
    """Privilégie T2 haute résolution, puis T2 star rapide, puis le premier scan."""
    scans = list(all_scans.items())
    normalized_protocols = []
    for _, scan_info in scans:
        protocol = unicodedata.normalize(
            "NFKD",
            str(scan_info.get("Protocol", ""))
            .casefold()
            .replace("*", "star"),
        )
        normalized_protocols.append(
            "".join(
                character
                for character in protocol
                if character.isalnum()
                and not unicodedata.combining(character)
            )
        )

    for index, protocol in enumerate(normalized_protocols):
        if "t2hauteresolution" in protocol:
            return index
    for index, protocol in enumerate(normalized_protocols):
        if "t2star" in protocol and "rapide" in protocol:
            return index
    return 0


def model_scan_index(
    all_scans: dict,
    model_path: pathlib.Path | None,
    model_species: str | None = None,
) -> int:
    """Sélectionne le canal IRM correspondant au modèle choisi."""
    if model_species == "rat":
        expected = ("t2turborare",)
        for index, (_, scan_info) in enumerate(all_scans.items()):
            protocol = unicodedata.normalize(
                "NFKD",
                str(scan_info.get("Protocol", ""))
                .casefold()
                .replace("*", "star"),
            )
            protocol = "".join(
                character
                for character in protocol
                if character.isalnum()
                and not unicodedata.combining(character)
            )
            if "t2turborare" in protocol:
                return index
        return default_scan_index(all_scans)

    if model_path is None:
        return default_scan_index(all_scans)

    if "rat" in str(model_path).casefold():
        expected = ("t2turborare",)
    elif "hem" in str(model_path).casefold():
        expected = ("t2starrapide", "t2star")
    elif "souris" in str(model_path).casefold():
        expected = ("t2hauteresolution",)
    else:
        return default_scan_index(all_scans)

    for index, (_, scan_info) in enumerate(all_scans.items()):
        protocol = unicodedata.normalize(
            "NFKD",
            str(scan_info.get("Protocol", ""))
            .casefold()
            .replace("*", "star"),
        )
        protocol = "".join(
            character
            for character in protocol
            if character.isalnum()
            and not unicodedata.combining(character)
        )
        if any(pattern in protocol for pattern in expected):
            return index

    return default_scan_index(all_scans)


def model_scan_id(
    all_scans: dict,
    model_path: pathlib.Path | None,
    model_species: str | None = None,
):
    """Retourne l'identifiant Bruker du scan choisi pour un modèle."""
    scan_index = model_scan_index(all_scans, model_path, model_species)
    scan_ids = list(all_scans)
    if 0 <= scan_index < len(scan_ids):
        return scan_ids[scan_index]
    return None


def convert_pvdataset_to_npy(file_path: str, scan_id) -> np.ndarray:
    """Convertit le scan sélectionné au format modèle (D, H, W)."""
    loader = brkraw.load(file_path)
    scans = loader.info(scope="scan", as_dict=True)
    if scan_id not in scans:
        raise ValueError(f"Analyse sélectionnée introuvable : {scan_id}.")

    raw_volume = np.asarray(loader.get_dataobj(scan_id))
    raw_volume = np.squeeze(raw_volume)
    while raw_volume.ndim > 3:
        raw_volume = raw_volume[..., 0]
    if raw_volume.ndim != 3:
        raise ValueError(
            f"L'analyse sélectionnée doit être un volume 3D, reçu {raw_volume.shape}."
        )

    # Les volumes Bruker sont généralement renvoyés en (H, W, D) ;
    # le modèle entraîné utilise le format (D, H, W) produit par loader.py.
    
    if ((raw_volume.shape[0] == 256 and raw_volume.shape[1] == 256) or 
       (raw_volume.shape[0] == 266 and raw_volume.shape[1] == 226)):
        
        model_volume = np.transpose(raw_volume, (2, 1, 0))
    
    
    elif ((raw_volume.shape[1] == 256 and raw_volume.shape[2] == 256) or
         (raw_volume.shape[1] == 226 and raw_volume.shape[2] == 266)):
        
        model_volume = raw_volume
         
    else:
        raise ValueError(
            "Dimensions incompatibles avec le 3D U-Net : "
            f"{raw_volume.shape}. Le modèle attend des coupes 256x256."
        )

    return model_volume.astype(np.float32, copy=False)


def prepare_volume(data: np.ndarray) -> np.ndarray:
    """Prépare un tableau 3D pour la navigation entre les coupes."""
    data = np.squeeze(data)

    while data.ndim > 3:
        data = data[..., 0]

    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    if data.ndim != 3:
        raise ValueError(
            f"Unsupported MRI array shape: {data.shape}"
        )

    return data
