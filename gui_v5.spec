from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    get_package_paths,
)


project_root = Path(SPECPATH)
gui_directory = project_root / "src" / "projet_master" / "GUI_APP"
asset_directory = gui_directory / "assets"
inference_module = project_root / "src" / "projet_master" / "model_inference.py"
brkraw_datas = collect_data_files("brkraw")
brkraw_hiddenimports = collect_submodules("brkraw")
_, brkraw_package_directory = get_package_paths("brkraw")
brkraw_info_directory = Path(brkraw_package_directory) / "apps" / "loader" / "info"
brkraw_datas.append(
    (
        str(brkraw_info_directory / "transform.py"),
        "brkraw/apps/loader/info",
    )
)
roifile_hiddenimports = collect_submodules("roifile")
gui_hiddenimports = (
    brkraw_hiddenimports
    + roifile_hiddenimports
    + [
        "projet_master.model_inference",
        "cv2",
        "pandas",
    ]
)
training_only_excludes = [
    "matplotlib",
    "networks",
    "pytorch_lightning",
    "segmentation_models_pytorch",
    "timm",
]


a = Analysis(
    [str(gui_directory / "app.py")],
    pathex=[str(project_root / "src")],
    datas=brkraw_datas
    + [
        (str(asset_directory), "projet_master/GUI_APP/assets"),
        (str(inference_module), "projet_master"),
    ],
    hiddenimports=gui_hiddenimports,
    excludes=training_only_excludes,
    hookspath=[str(project_root / "pyinstaller_hooks")],
    noarchive=False,
)


pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STR@KE ALLIANCE MRI VIEWER",
    console=False,
    icon=str(asset_directory / "app_icon.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="STR@KE ALLIANCE MRI VIEWER",
)
