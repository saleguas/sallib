# src/saltools/resources.py  (minimal version)

from __future__ import annotations
import os, platform, urllib.request, shutil
from pathlib import Path
import appdirs       # pip install appdirs>=1.4

CACHE = Path(os.getenv("SALTOOLS_CACHE_DIR",
                       appdirs.user_cache_dir("saltools", "sal")))
CACHE.mkdir(parents=True, exist_ok=True)

# ---------- registry --------------------------------------------------------
# Only one entry for now — add more later if you want.
_REGISTRY = {
    "magick": {
        ("Windows", "AMD64"): {
            "url": "https://imagemagick.org/archive/binaries/"
                   "ImageMagick-7.1.1-47-Q16-HDRI-x64-dll.exe",
            "filename": "magick.exe",     # how we'll store it locally
        }
    }
}
# ---------------------------------------------------------------------------

def gb(name: str) -> Path:
    """Download the binary if needed, then return its Path."""
    # 1️⃣ env-var override
    env = os.getenv(f"SALTOOLS_{name.upper()}")
    if env:
        return Path(env).expanduser()

    # 2️⃣ already on PATH?
    if (p := shutil.which(name)):
        return Path(p)

    # 3️⃣ look up registry entry for this OS/arch
    sys_id = (platform.system(), platform.machine())
    try:
        entry = (_REGISTRY[name].get(sys_id)
                 or _REGISTRY[name].get((sys_id[0], "*")))   # allow "any arch"
    except KeyError:
        raise RuntimeError(f"No registry entry for {name!r}")

    # 4️⃣ cached already?
    dest = CACHE / name / entry["filename"]
    if dest.exists():
        return dest

    # 5️⃣ download
    url = entry["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"→ Downloading {url}")
    urllib.request.urlretrieve(url, tmp)

    # 6️⃣ finalize
    tmp.rename(dest)
    dest.chmod(dest.stat().st_mode | 0o111)      # make sure it's executable
    return dest
