"""Motor de segmentação local e correções de máscara do FundoZero."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

EXTENSOES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
MODELOS = {
    'Equilibrado | Fotos e objetos': 'isnet-general-use',
    'Personagens | Anime e ilustrações': 'isnet-anime',
    'Rápido | Modelo compacto': 'u2netp',
    'Detalhado | Recortes complexos': 'birefnet-general-lite',
    'Retratos | Pessoas e cabelo': 'birefnet-portrait',
}


def abrir_imagem(arquivo: str | Path) -> Image.Image:
    with Image.open(arquivo) as entrada:
        entrada.seek(0)  # GIF: primeiro quadro
        original = ImageOps.exif_transpose(entrada)
        return original.convert('RGBA')


def remover_fundo(original: Image.Image, sessao, descontaminar: bool = True) -> Image.Image:
    """Executa inferência com rembg inteiramente na máquina local."""
    from rembg import remove

    resultado = remove(original.convert('RGB'), session=sessao, decontaminate=descontaminar)
    if not isinstance(resultado, Image.Image):
        from io import BytesIO
        resultado = Image.open(BytesIO(resultado)).convert('RGBA')
    else:
        resultado = resultado.convert('RGBA')
    if resultado.size != original.size:
        resultado = resultado.resize(original.size, Image.Resampling.LANCZOS)
    alpha_original = original.getchannel('A')
    if alpha_original.getextrema() != (255, 255):
        resultado.putalpha(ImageChops.multiply(resultado.getchannel('A'), alpha_original))
    return resultado


@dataclass(frozen=True)
class Traco:
    modo: str  # 'apagar' ou 'restaurar'
    diametro: int
    pontos: tuple[tuple[int, int], ...]


def _pincelar(desenho: ImageDraw.ImageDraw, pontos: tuple[tuple[int, int], ...], raio: int, cor: int) -> None:
    if not pontos:
        return
    largura = raio * 2
    if len(pontos) > 1:
        desenho.line(pontos, fill=cor, width=largura, joint='curve')
    for x, y in (pontos if len(pontos) < 2 else (pontos[0], pontos[-1])):
        desenho.ellipse((x - raio, y - raio, x + raio, y + raio), fill=cor)


def construir_pinceis(tamanho: tuple[int, int], tracos: Iterable[Traco]) -> tuple[Image.Image, Image.Image]:
    apagar = Image.new('L', tamanho, 0)
    restaurar = Image.new('L', tamanho, 0)
    pa = ImageDraw.Draw(apagar)
    pr = ImageDraw.Draw(restaurar)
    for traco in tracos:
        raio = max(1, traco.diametro // 2)
        if traco.modo == 'apagar':
            _pincelar(pa, traco.pontos, raio, 255)
            _pincelar(pr, traco.pontos, raio, 0)
        else:
            _pincelar(pr, traco.pontos, raio, 255)
            _pincelar(pa, traco.pontos, raio, 0)
    return apagar, restaurar


def finalizar_imagem(
    original: Image.Image,
    recorte: Image.Image,
    borda: int = 0,
    suavidade: int = 0,
    limpeza: int = 0,
    tracos: Iterable[Traco] = (),
) -> Image.Image:
    """Ajusta bordas e restaura regiões pintadas sem reinvocar a IA."""
    if original.size != recorte.size:
        raise ValueError('O recorte e a imagem original devem ter o mesmo tamanho.')
    alpha = recorte.getchannel('A')
    borda = max(-8, min(8, int(borda)))
    if borda > 0:
        alpha = alpha.filter(ImageFilter.MaxFilter(2 * borda + 1))
    elif borda < 0:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * abs(borda) + 1))
    suavidade = max(0, min(12, int(suavidade)))
    if suavidade:
        alpha = alpha.filter(ImageFilter.GaussianBlur(suavidade / 2))
    limpeza = max(0, min(180, int(limpeza)))
    if limpeza:
        # Remove transparências residuais fracas sem destruir áreas opacas.
        lut = [0 if x <= limpeza else min(255, round((x - limpeza) * 255 / (255 - limpeza))) for x in range(256)]
        alpha = alpha.point(lut)

    tracos = tuple(tracos)
    if tracos:
        apagar, restaurar = construir_pinceis(original.size, tracos)
        alpha = ImageChops.subtract(alpha, apagar)
        alpha = ImageChops.lighter(alpha, restaurar)
        rgb = Image.composite(original.convert('RGB'), recorte.convert('RGB'), restaurar)
    else:
        rgb = recorte.convert('RGB')
    resultado = rgb.convert('RGBA')
    resultado.putalpha(ImageChops.multiply(alpha, original.getchannel('A')))
    return resultado


def imagem_ajustada(entrada: str | Path, saida: str | Path, sessao, 
                     descontaminar: bool, borda: int, suavidade: int, limpeza: int) -> None:
    original = abrir_imagem(entrada)
    recorte = remover_fundo(original, sessao, descontaminar)
    final = finalizar_imagem(original, recorte, borda, suavidade, limpeza)
    final.save(saida, format='PNG', optimize=True)
