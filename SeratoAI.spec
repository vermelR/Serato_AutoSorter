# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('app.py', '.'), ('phase4_engine.py', '.'), ('phase1_harvest.py', '.'), ('serato_writer.py', '.'), ('config.py', '.'), ('identify.py', '.'), ('genre_model.py', '.'), ('tag_writer.py', '.'), ('watcher.py', '.'), ('serato_model.pkl', '.'), ('music_features_dataset.csv', '.'), ('vendor/ffmpeg', 'vendor')],
    hiddenimports=['streamlit', 'streamlit.web', 'streamlit.web.cli', 'streamlit.runtime', 'sklearn', 'librosa', 'soundfile', 'audioread', 'pyserato', 'acoustid', 'musicbrainzngs', 'mutagen'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SeratoAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SeratoAI',
)
app = BUNDLE(
    coll,
    name='SeratoAI.app',
    icon=None,
    bundle_identifier=None,
)
