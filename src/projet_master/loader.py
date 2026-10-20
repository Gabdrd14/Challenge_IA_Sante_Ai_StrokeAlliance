import zipfile, pathlib, tempfile, re, csv
import numpy as np
import cv2
from roifile import roiread
from brukerapi.dataset import Dataset

RACINE = pathlib.Path("FULLMRI")
SORTIE = pathlib.Path("DATA_AVC_NPY")
LIMITE = None         # nombre max de paires par dossier pour un premier test ; None = tout convertir
A_ECARTER = set()   # (étude, souris, jour)

ETUDES = {"Etude_3"} # Pour choisir les études à convertir

def souris(nom):
    m = re.search(r"M0*(\d+)(?!\d)", nom)
    return int(m.group(1)) if m else None

def protocole(scan):
    acqp = scan / "acqp"
    if not acqp.exists():
        return ""
    m = re.search(r"##\$ACQ_protocol_name=.*\n<([^>]*)>", acqp.read_text(errors="ignore"))
    return m.group(1) if m else ""

def date_acquisition(fichier):
    with zipfile.ZipFile(fichier) as z:
        return z.namelist()[0].split("/")[0][:15]          # ex. 20230620_090756

def lire_t2(fichier):
    """Volume (20, 256, 256) du T2 haute résolution, ou (None, raison)."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(fichier) as z:
            z.extractall(tmp)
        trouves = [p for p in pathlib.Path(tmp).rglob("2dseq")
                   if "T2_haute_resolution" in protocole(p.parents[2])]
        if len(trouves) != 1:
            return None, f"{len(trouves)} scan(s) T2 haute résolution"
        arr = Dataset(trouves[0]).data
        if arr.shape != (256, 256, 20):
            return None, f"forme {arr.shape}"
        return np.transpose(arr, (2, 1, 0)).astype(np.float32), ""     # orientation validée

def construire_masque(fichier_roi, shape):
    n, H, W = shape
    masque = np.zeros(shape, np.uint8) # volume rempli de 0 (fond)
    nb, ignorees = 0, []
    for r in roiread(fichier_roi):
        z = (r.position if r.position > 0 else int(r.name.split("-")[0])) - 1
        pts = r.coordinates()
        if pts is None or not (0 <= z < n):
            ignorees.append(r.name)
            continue
        cv2.fillPoly(masque[z], [np.round(pts).astype(np.int32)], 1) # intérieur du polygone = 1 (ROI)
        nb += 1
    return masque, nb, ignorees

# 1. Inventaire : étude / jour / souris -> fichiers IRM et ROI 
irm, roi = {}, {}
for d in RACINE.rglob("*"):
    m = re.fullmatch(r"(IRM|ROI)\s*(D\d+)", d.name.strip()) if d.is_dir() else None
    if not m:
        continue
    type_, jour = m.groups()
    etude = next((p.name for p in d.parents if re.fullmatch(r"Etude_\d+", p.name)), None)

    if etude not in ETUDES:
        continue

    cible = irm if type_ == "IRM" else roi
    for f in d.iterdir():
        s = souris(f.name)
        if f.name.startswith("._") or s is None:
            continue
        if (type_ == "IRM" and f.suffix == ".PvDatasets") or (type_ == "ROI" and f.suffix.lower() == ".zip"):
            cible.setdefault((etude, jour, s), []).append(f)

paires = sorted(k for k in irm if k in roi and k not in {(e, j, s) for e, s, j in A_ECARTER})
print(len(paires), "paires IRM+ROI trouvées")

# 2. Conversion
SORTIE.mkdir(exist_ok=True)
compteur = {}
with open(f"suivi_conversion_{'_'.join(sorted(ETUDES))}.csv", "w", newline="", encoding="utf-8") as fh:

    w = csv.writer(fh, delimiter=";")
    w.writerow(["etude", "jour", "souris", "statut", "nb_roi", "coupes_lesion",
                "pixels_lesion", "irm_gardee", "nb_irm", "remarque"])
    for (etude, jour, s) in paires:
        dossier = (etude, jour)
        compteur[dossier] = compteur.get(dossier, 0) + 1
        if LIMITE and compteur[dossier] > LIMITE:
            continue
        candidats, rois = irm[(etude, jour, s)], roi[(etude, jour, s)]
        if len(rois) != 1:
            w.writerow([etude, jour, s, "ecarte", "", "", "", "", len(candidats), f"{len(rois)} fichiers ROI"])
            continue
        gardee = sorted(candidats, key=date_acquisition)[0]          # le plus ancien
        try:
            vol, err = lire_t2(gardee)
            if vol is None:
                w.writerow([etude, jour, s, "ecarte", "", "", "", gardee.name[:40], len(candidats), err])
                continue
            masque, nb, ign = construire_masque(rois[0], vol.shape)
            out = SORTIE / etude / jour
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"M{s:03d}_img.npy", vol) # sauvegarde du volume T2
            np.save(out / f"M{s:03d}_mask.npy", masque) # sauvegarde du masque
            w.writerow([etude, jour, s, "ok", nb, int((masque.sum((1, 2)) > 0).sum()),
                        int(masque.sum()), gardee.name[:40], len(candidats),
                        f"ROI ignorées : {ign}" if ign else ""])
            print(etude, jour, f"M{s:03d}", "ok")
        except Exception as e:
            w.writerow([etude, jour, s, "erreur", "", "", "", gardee.name[:40], len(candidats), str(e)[:100]])
            print(etude, jour, f"M{s:03d}", "erreur :", e)