# Stroke Alliance MRI Viewer

A desktop application for browsing Bruker `.PvDatasets`, viewing MRI slices,
running segmentation models, and editing or exporting ImageJ ROI files.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A directory containing model checkpoints (`.ckpt`) for segmentation

## Run

From the repository root, install the project dependencies and start the app:

```powershell
uv sync
uv run python src\projet_master\GUI_APP\app.py
```

In VS Code, select the project's `.venv` Python interpreter and run
`src\projet_master\GUI_APP\app.py` directly.

## Build an executable

Use the project spec to bundle the app, its assets, and Bruker resources:

```powershell
uv run pyinstaller --clean --noconfirm gui_v5.spec
```

The application is created under `dist\STR@KE ALLIANCE MRI VIEWER\`. Select a
model checkpoint directory separately when using it.

## Use

1. Select a folder containing `.PvDatasets` files.
2. Select a model directory containing at least one `.ckpt` file.
3. Open a dataset, select its scan and model, and run the analysis.
4. Browse slices, adjust their display, edit the predicted mask, and save
   contours as an ImageJ ROI ZIP. Existing ROI ZIPs can also be opened.

The app also offers analysis of all datasets in a selected folder. Model
directory and mask-drawing preferences are saved between launches.

## App structure

The application lives in `src/projet_master/GUI_APP/`. `app.py` is the entry
point; the other modules handle the windows, image display, datasets,
predictions, ROI files, and settings. Logos and icons are in `assets/`.
