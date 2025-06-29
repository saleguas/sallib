
import os
from pathlib import Path
import subprocess
import shlex
from sallib.resources import gb

# get the path to the magick binary
MAGICK = gb("magick")

def _run(cmd: str) -> str:
    """Run a shell command and return its stdout."""
    return subprocess.check_output(shlex.split(cmd), text=True).strip()


def resize_image_to_target(image_path, target_w=940, target_h=1250, out_path=None):
    """
    Resize an image to target dimensions while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the input image
        target_w (int): Target width (default: 940)
        target_h (int): Target height (default: 1250)
        out_path (str, optional): Output path. If None, saves to OUTPUT_PATH
        
    Returns:
        str: Path to the resized image
    """
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Get original dimensions
    overlay_w = int(
        _run(f'{MAGICK} identify -format "%w" {shlex.quote(str(src))}')
    )
    overlay_h = int(
        _run(f'{MAGICK} identify -format "%h" {shlex.quote(str(src))}')
    )
    
    # Calculate ratios to determine which dimension is closer to target
    width_ratio = overlay_w / target_w
    height_ratio = overlay_h / target_h
    
    # Pick the dimension that's closer to the target (ratio closer to 1.0)
    if abs(width_ratio - 1.0) <= abs(height_ratio - 1.0):
        # Width is closer to target, scale based on width
        scale_factor = target_w / overlay_w
        new_w = target_w
        new_h = int(overlay_h * scale_factor)
        print(f"Scaling based on width: {overlay_w}x{overlay_h} -> {new_w}x{new_h}")
    else:
        # Height is closer to target, scale based on height  
        scale_factor = target_h / overlay_h
        new_h = target_h
        new_w = int(overlay_w * scale_factor)
        print(f"Scaling based on height: {overlay_w}x{overlay_h} -> {new_w}x{new_h}")

    # if new_w is greater than 940, now scale both down until width is 940
    if new_w > target_w:
        scale_factor = target_w / new_w
        new_w = target_w
        new_h = int(new_h * scale_factor)
        print(f"Scaling both down: {overlay_w}x{overlay_h} -> {new_w}x{new_h}")
    
    # Resize the image
    _run(f'{MAGICK} {shlex.quote(str(src))} -resize {new_w}x{new_h}! {shlex.quote(out_path)}')
    
    return out_path