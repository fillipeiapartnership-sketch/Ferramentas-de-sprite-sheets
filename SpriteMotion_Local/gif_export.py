"""Animated transparent GIF export from the same corrected RGBA PNGs used by atlas.

GIF inherently has a 256-entry palette and 1-bit transparency, so PNGs remain
primary game assets. Transparent index is reserved (0); opaque indexes 1-255.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from sprite_engine import geometry, render_cell


def indexed_gif_frame(rgba: Image.Image) -> Image.Image:
    """Per-frame palette, indexed transparent pixel 0 and at most 255 opaque colors."""
    rgba = rgba.convert('RGBA')
    alpha = np.asarray(rgba.getchannel('A'))
    # GIF cannot represent soft alpha edges: make the least damaging binary split.
    visible = alpha >= 128
    # RGB in transparent pixels is irrelevant. Remove it before quantization.
    rgb = np.asarray(rgba.convert('RGB')).copy()
    rgb[~visible] = (0, 0, 0)
    quantized = Image.fromarray(rgb, 'RGB').quantize(
        colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    colors = np.asarray(quantized, dtype=np.uint8)
    # Quantizer uses 0..254; reserve index 0 for transparent pixels.
    indexed = np.zeros(alpha.shape, dtype=np.uint8)
    indexed[visible] = (colors[visible].astype(np.uint16) + 1).astype(np.uint8)
    result = Image.fromarray(indexed, mode='P')
    old_palette = quantized.getpalette() or []
    old_palette = (old_palette + [0] * 765)[:765]
    result.putpalette([0, 0, 0] + old_palette)
    result.info['transparency'] = 0
    return result


def export_animation_gif(group, dest: str | Path, indexes: list[int],
                         fps: int, cellw: int, cellh: int,
                         stabilize: bool = False, auto_fit: bool = True,
                         progress=None, cancelled=None) -> Path:
    """Uses selected edited PNG sprites, fixed geometry, playback order and FPS."""
    if not indexes:
        raise ValueError('Selecione sprites para criar o GIF.')
    if any(not group.frames[i].sprite or not Path(group.frames[i].sprite).is_file() for i in indexes):
        raise ValueError('Alguns quadros selecionados ainda não têm PNG transparente.')
    if len(indexes) * cellw * cellh > 48_000_000:
        raise ValueError('GIF muito grande para a memória. Reduza tamanho da grid ou número de quadros.')
    fps = max(1, min(60, int(fps)))
    duration = max(20, round(1000/fps))  # GIF duration resolution is 10 milliseconds
    paths = [group.frames[i].sprite for i in indexes]
    lay = geometry(paths, cellw, cellh, stabilize, auto_fit)
    frames = []
    for n, idx in enumerate(indexes):
        if cancelled is not None and cancelled():
            raise InterruptedError('Exportação do GIF cancelada.')
        corrected = render_cell(group.frames[idx].sprite, lay)
        frames.append(indexed_gif_frame(corrected))
        if progress is not None:
            progress(round(95*(n+1)/len(indexes)), f'Preparando GIF: {n+1}/{len(indexes)}')
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix('.tmp.gif')
    try:
        if progress is not None:
            progress(96, 'Gravando GIF no disco. Aguarde a compactação...')
        frames[0].save(
            temp_path, format='GIF', save_all=True, append_images=frames[1:],
            duration=duration, loop=0, transparency=0, disposal=2, optimize=False
        )
        if cancelled is not None and cancelled():
            raise InterruptedError('Exportação do GIF cancelada.')
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    if progress is not None:
        progress(100, f'GIF gerado: {destination.name}')
    return destination
