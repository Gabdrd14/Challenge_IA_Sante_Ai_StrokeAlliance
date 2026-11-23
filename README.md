## Build the GUI executable

Build `gui_v5` with the project spec so Bruker's `brkraw` YAML resources are
included in the executable:

```powershell
uv run pyinstaller --clean --noconfirm gui_v5.spec
```

Running PyInstaller directly on `gui_v5.py` omits `brkraw/apps/loader/info/*.yaml`
and causes a `scan.yaml` loading error at runtime.
