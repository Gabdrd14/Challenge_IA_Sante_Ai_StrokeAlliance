import os

from PySide6.QtCore import QDateTime


def file_generator(folder_path: str , output_folder: str ,nb_slice : int) -> None:
    """Générateur du fichier excel des résultats sur l'ensemble des sujets du dossier racine """
    name_file = f"results_{folder_path}_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.xlsx"
    """Structure du fichier excel : id , surface slice , surface slice +1, surface slice + n , volume total"""
    
    """fonction parcoura chaque roifile dans le dossier output de la prediction totale du dossier"""
    # Construire dynamiquement les colonnes de surface :
    # - une colonne "surface_slice_i" pour chaque coupe comprise entre 1 et nb_slice
    # - conserver "id" comme première colonne
    # - conserver "volume_total" comme dernière colonne
    # colonnes_slice = [f"surface_slice_{i + 1}" for i in range(nb_slice)]
    # colonnes = ["id", *colonnes_slice, "volume_total"]
    # dataframe = pd.DataFrame(columns=colonnes)

    # Parcourir chaque fichier ROI du dossier contenant les prédictions totales.
    # Pour chaque ROI :
    # - identifier le sujet dans le nom du fichier ou son dossier parent ;
    # - lire les coordonnées des polygones et leur position de coupe ;
    # - calculer la surface de chaque coupe ;
    # - ajouter une ligne du tableau pour le sujet ;
    # - sommer les surfaces des coupes pour obtenir le volume total.
    # dossier_pred_total = pathlib.Path(output_folder)
    # roi_files = sorted(dossier_pred_total.rglob("*.zip"))
    # for roi_file in roi_files:
    #     ...


    if os.path.exists(f"{output_folder}/{name_file}"):
        i = 1
        while True:
            new_name_file = f"results_{folder_path}_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}_{i}.xlsx"
            if not os.path.exists(new_name_file):
                name_file = new_name_file
                break
            i += 1
