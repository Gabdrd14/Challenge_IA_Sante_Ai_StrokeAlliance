import zipfile
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from roifile import ImagejRoi


def browse_roi_zip(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:
        roi_files = [
            f for f in z.namelist()
            if f.lower().endswith(".roi")
        ]

    if not roi_files:
        print("Aucun fichier ROI trouvé.")
        return

    index = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    def display_roi():

        ax.clear()

        filename = roi_files[index]

        with zipfile.ZipFile(zip_path, "r") as z:
            roi_data = z.read(filename)

        roi = ImagejRoi.frombytes(roi_data)

        # Coordonnées des points de la ROI
        coords = roi.coordinates()

        if coords is not None and len(coords) > 0:
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                "-",
                linewidth=2
            )
            ax.scatter(coords[:, 0], coords[:, 1], s=5)

            ax.invert_yaxis()
            ax.set_aspect("equal")

        ax.set_title(
            f"{filename}\n"
            f"ROI {index + 1}/{len(roi_files)}"
        )

        ax.grid(True)
        fig.canvas.draw_idle()

    def next_roi(event=None):
        nonlocal index
        index = (index + 1) % len(roi_files)
        display_roi()

    def previous_roi(event=None):
        nonlocal index
        index = (index - 1) % len(roi_files)
        display_roi()

    # Boutons
    ax_prev = plt.axes([0.25, 0.05, 0.2, 0.08])
    ax_next = plt.axes([0.55, 0.05, 0.2, 0.08])

    btn_prev = Button(ax_prev, "Previous")
    btn_next = Button(ax_next, "Next")

    btn_prev.on_clicked(previous_roi)
    btn_next.on_clicked(next_roi)

    def on_key(event):
        if event.key == "right":
            next_roi()
        elif event.key == "left":
            previous_roi()

    fig.canvas.mpl_connect("key_press_event", on_key)

    display_roi()
    plt.show()
    
browse_roi_zip("/Users/gabriel/Documents/MRI/projet_master/test/CC177_P02_0825_Thrombin_V2_M128_D1_CC177_P02_0825_Thrombin_V2_M128_D1_17630173_1_Default_D1_17630201_6.0.1.zip")  # Remplacez par le chemin de votre fichier ZIP