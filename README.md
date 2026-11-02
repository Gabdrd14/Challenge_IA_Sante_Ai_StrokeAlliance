## Build the GUI executable

Build `guiv2` with the project spec so Bruker's `brkraw` YAML resources are
included in the executable:

```powershell
uv run pyinstaller --clean --noconfirm guiv2.spec
```

Running PyInstaller directly on `guiv2.py` omits `brkraw/apps/loader/info/*.yaml`
and causes a `scan.yaml` loading error at runtime.