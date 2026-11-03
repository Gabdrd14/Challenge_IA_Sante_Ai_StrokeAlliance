from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    get_package_paths,
)


project_root = Path(SPECPATH)
gui_directory = project_root / "src" / "projet_master" / "GUI"
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


a = Analysis(
    [str(gui_directory / "guiv2.py")],
    pathex=[str(project_root / "src")],
    datas=brkraw_datas
    + [
        (str(gui_directory / "logo.png"), "."),
        (str(gui_directory / "etaplablogo.png"), "."),
        (str(inference_module), "projet_master"),
    ],
    hiddenimports=brkraw_hiddenimports + ["projet_master.model_inference"],
    hookspath=[str(project_root / "pyinstaller_hooks")],
    noarchive=False,
)
splash = Splash(
    str(gui_directory / "logo.png"),
    binaries=a.binaries,
    datas=a.datas,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STR@KE ALLIANCE MRI VIEWER",
    console=False,
    icon=str(gui_directory / "app_icon.ico"),
    splash=splash,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="STR@KE ALLIANCE MRI VIEWER",
)
