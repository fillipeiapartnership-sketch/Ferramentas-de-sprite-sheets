"""Retoque RGBA offline em duas fases: selecionar, visualizar, confirmar.

O clique não modifica o sprite. Uma seleção conexa delimitada por cor,
raio e tamanho máximo só é aplicada após confirmação explícita.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class RetouchSelection:
    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    action: str
    count: int
    source_shape: tuple[int, int]


def _near_boundary(alpha: np.ndarray, x: int, y: int) -> float:
    opaque = (alpha > 12).astype(np.uint8)
    if not opaque[y, x]:
        return 0.0
    if np.all(opaque):
        return float(min(x + 1, y + 1, alpha.shape[1] - x, alpha.shape[0] - y))
    return float(cv2.distanceTransform(opaque, cv2.DIST_L2, 3)[y, x])


def _selected_patch(rgba: np.ndarray, x: int, y: int, tolerance: int, radius: int):
    height, width = rgba.shape[:2]
    r = max(2, min(int(radius), 160))
    x0, y0 = max(0, x-r), max(0, y-r)
    x1, y1 = min(width, x+r+1), min(height, y+r+1)
    patch = rgba[y0:y1, x0:x1]
    sx, sy = x-x0, y-y0
    rgb = np.ascontiguousarray(patch[:, :, :3])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed = lab[sy, sx].copy()
    distance = np.sqrt(np.sum((lab - seed)**2, axis=2))
    acceptable = (distance <= max(2, int(tolerance)) * 1.15) & (patch[:, :, 3] > 5)
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    acceptable &= ((xx-sx)**2 + (yy-sy)**2 <= r*r)
    if not acceptable[sy, sx]:
        return None
    _, labels = cv2.connectedComponents(acceptable.astype(np.uint8), connectivity=8)
    component = int(labels[sy, sx])
    if component <= 0:
        return None
    return (x0, y0, x1, y1, labels == component)


def select_retouch(rgba: np.ndarray, x: int, y: int, tolerance: int = 19,
                   radius: int = 28, mode: str = 'auto') -> tuple[RetouchSelection | None, str]:
    """Seleciona a área sem modificar nenhum pixel do RGBA fornecido."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError('A imagem do editor precisa estar em RGBA de 8 bits.')
    h, w = rgba.shape[:2]
    if not (0 <= int(x) < w and 0 <= int(y) < h):
        return None, 'Clique fora da imagem.'
    x, y = int(x), int(y)
    if rgba[y, x, 3] <= 5:
        return None, 'Esta região já está transparente. Clique sobre a mancha.'
    region = _selected_patch(rgba, x, y, tolerance, radius)
    if region is None:
        return None, 'Nenhuma mancha foi encontrada neste ponto.'
    x0, y0, x1, y1, picked = region
    total = int(np.count_nonzero(picked))
    if total < 1:
        return None, 'Nenhuma mancha foi encontrada.'
    r = max(2, min(int(radius), 160))
    if total > min(12000, max(180, round(np.pi*r*r*.46))):
        return None, ('Seleção muito grande. Reduza a tolerância ou o raio '
                      'para proteger a roupa e a pele do personagem.')
    rgb = rgba[y, x, :3].astype(int)
    green_cast = int(rgb[1]) >= int(max(rgb[0], rgb[2])) + 11 and int(rgb[1]) >= 45
    near = _near_boundary(rgba[:, :, 3], x, y)
    translucent = int(rgba[y, x, 3]) < 220
    if mode == 'auto':
        action = 'erase' if (near <= 4.5 and (green_cast or translucent)) else 'color'
    elif mode in ('erase', 'color'):
        action = mode
    else:
        raise ValueError('Modo de retoque inválido.')
    selection = RetouchSelection(
        bounds=(x0, y0, x1, y1), mask=picked.copy(), action=action,
        count=total, source_shape=(h, w))
    action_label = 'apagados para transparência' if action == 'erase' else 'com a cor corrigida'
    return selection, f'{total} pixels selecionados, que serão {action_label}. Confira a área destacada e confirme.'


def apply_retouch_selection(rgba: np.ndarray, selection: RetouchSelection) -> tuple[np.ndarray, str, int]:
    """Aplica uma seleção já aprovada, deixando o RGBA de entrada intacto."""
    if selection.source_shape != tuple(rgba.shape[:2]):
        raise ValueError('As dimensões do sprite mudaram após a seleção.')
    x0, y0, x1, y1 = selection.bounds
    picked = selection.mask
    if picked.shape != (y1-y0, x1-x0):
        raise ValueError('Máscara de seleção incompatível.')
    out = rgba.copy()
    total = selection.count
    if selection.action == 'erase':
        target = out[y0:y1, x0:x1]
        target[:, :, 3][picked] = 0
        target[:, :, :3][picked] = 0
        message = f'{total} pixels apagados para transparência após confirmação.'
    elif selection.action == 'color':
        pad = 5
        h, w = rgba.shape[:2]
        ax0, ay0 = max(0, x0-pad), max(0, y0-pad)
        ax1, ay1 = min(w, x1+pad), min(h, y1+pad)
        crop = np.ascontiguousarray(out[ay0:ay1, ax0:ax1, :3].copy())
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[y0-ay0:y1-ay0, x0-ax0:x1-ax0][picked] = 255
        transparent = (out[ay0:ay1, ax0:ax1, 3] <= 5)
        if transparent.any():
            mask[transparent] = 255
        repaired = cv2.inpaint(crop, mask, 2.0, cv2.INPAINT_TELEA)
        rgbtarget = out[y0:y1, x0:x1, :3]
        rgbtarget[picked] = repaired[y0-ay0:y1-ay0, x0-ax0:x1-ax0][picked]
        message = f'{total} pixels de cor corrigidos após confirmação; transparência preservada.'
    else:
        raise ValueError('Ação de retoque inválida.')
    return out, message, total


def retouch_click(rgba: np.ndarray, x: int, y: int, tolerance: int = 19,
                  radius: int = 28, mode: str = 'auto') -> tuple[np.ndarray, str, int]:
    """Compatibilidade com testes antigos. Editor usa seleção + confirmação."""
    selected, message = select_retouch(rgba, x, y, tolerance, radius, mode)
    if selected is None:
        return rgba.copy(), message, 0
    return apply_retouch_selection(rgba, selected)
