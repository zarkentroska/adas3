# -*- mode: python ; coding: utf-8 -*-
"""
Especificación personalizada de PyInstaller para empaquetar testcam.py
incluyendo todos los recursos y modelos necesarios.

Uso:
    pyinstaller --noconfirm pyinstaller.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# __file__ no está definido cuando PyInstaller ejecuta el spec mediante exec(),
# así que usamos el directorio actual desde el que se invoca el comando.
BASE_DIR = os.path.abspath(os.getcwd())
ICON_PATH = os.path.join(BASE_DIR, "settings.png")

RESOURCE_FILES = [
    "best.pt",
    "drone_audio_model.h5",
    "audio_mean.npy",
    "audio_std.npy",
    "config_camara.json",
    "tinysa_advanced_intervals.json",
    "yolo_models_config.json",
    "settings.png",
    "__best.pt",
]

datas = []
for resource in RESOURCE_FILES:
    src = os.path.join(BASE_DIR, resource)
    if os.path.exists(src):
        datas.append((src, "."))

# Datos adicionales para bibliotecas que cargan archivos en tiempo de ejecución
datas += collect_data_files("matplotlib", include_py_files=True)
datas += collect_data_files("cv2")
datas += collect_data_files("librosa")
datas += collect_data_files("ultralytics")
datas += copy_metadata("ultralytics")
datas += copy_metadata("librosa")
datas += copy_metadata("tensorflow")

hiddenimports = []
for pkg in [
    "matplotlib",
    "cv2",
    "ultralytics",
    "librosa",
    "tensorflow",
    "pyaudio",
    "serial",
]:
    hiddenimports += collect_submodules(pkg)

binaries = []

block_cipher = None

a = Analysis(
    ["testcam.py"],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DetectorDrones",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH if os.path.exists(ICON_PATH) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DetectorDrones",
)

