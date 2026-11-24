"""Recorte local de GIFs e vídeos inteiros, preservando movimento e transparência.

GIF: transparência binária (limitação do formato GIF); WebP e WebM: alfa suave.
Vídeos: vídeo quadro a quadro em fluxo, sem manter o filme na RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import subprocess
import tempfile

from PIL import Image, ImageOps

from motor import finalizar_imagem, remover_fundo

EXTENSOES_VIDEO = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.wmv', '.mpeg', '.mpg'}
EXTENSOES_GIF = {'.gif'}
EXTENSOES_ESTATICAS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
EXTENSOES_SUPORTADAS = EXTENSOES_ESTATICAS | EXTENSOES_GIF | EXTENSOES_VIDEO


def tipo_arquivo(caminho: str | Path) -> str:
    ext = Path(caminho).suffix.lower()
    if ext in EXTENSOES_GIF:
        return 'gif'
    if ext in EXTENSOES_VIDEO:
        return 'video'
    if ext == '.webp':
        # WebP pode conter vários quadros; tratá-los como animação.
        with Image.open(caminho) as imagem:
            return 'gif' if getattr(imagem, 'n_frames', 1) > 1 else 'imagem'
    if ext in EXTENSOES_ESTATICAS:
        return 'imagem'
    raise ValueError(f'Formato não suportado: {ext}')


def primeira_imagem(caminho: str | Path) -> Image.Image:
    """Imagem de prévia, NÃO processa a animação inteira."""
    if tipo_arquivo(caminho) != 'video':
        with Image.open(caminho) as img:
            img.seek(0)
            return ImageOps.exif_transpose(img).convert('RGBA')
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(str(Path(caminho).resolve()), pix_fmt='rgb24')
    try:
        meta = next(frames)
        w, h = meta['size']
        buffer = next(frames)
        return Image.frombytes('RGB', (w, h), buffer).convert('RGBA')
    finally:
        frames.close()


def _recortar(original: Image.Image, sessao, descontaminar: bool, ajustes: dict) -> Image.Image:
    resultado = remover_fundo(original, sessao, descontaminar)
    return finalizar_imagem(original, resultado, **ajustes)


def _gif_paleta_transparente(frame: Image.Image) -> Image.Image:
    """Reserva índice 0 para transparência; nunca sacrifica o índice de uma cor real."""
    rgba = frame.convert('RGBA')
    rgb = rgba.convert('RGB')
    reduzida = rgb.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    paleta = reduzida.getpalette()[:255 * 3]
    paleta = [0, 0, 0] + paleta
    paleta += [0] * (768 - len(paleta))
    trocar = bytes.maketrans(bytes(range(256)), bytes(min(i + 1, 255) for i in range(256)))
    indices = reduzida.tobytes().translate(trocar)
    # GIF aceita apenas alfa 0 ou 255; pixels intermediários passam por limiar.
    opaco = rgba.getchannel('A').point(lambda a: 255 if a >= 128 else 0)
    final = Image.new('P', rgba.size, 0)
    final.putpalette(paleta)
    com_cores = Image.frombytes('P', rgba.size, indices)
    com_cores.putpalette(paleta)
    final.paste(com_cores, (0, 0), opaco)
    final.info['transparency'] = 0
    final.info['disposal'] = 2
    return final


def processar_gif(
    entrada: Path, saida: Path, sessao, descontaminar: bool, ajustes: dict,
    formato: str = 'gif', avisar: Callable[[str], None] | None = None,
) -> int:
    """Processa TODOS os quadros do GIF, mantendo duração de cada um e loop."""
    entrada, saida = Path(entrada), Path(saida)
    if formato not in ('gif', 'webp'):
        raise ValueError('Formato de animação inválido. Escolha GIF ou WebP.')
    quadros = []
    duracoes = []
    with Image.open(entrada) as animacao:
        loop = animacao.info.get('loop', 1)  # sem extensão NETSCAPE equivale a uma reprodução
        quantidade = getattr(animacao, 'n_frames', 1)
        for i in range(quantidade):
            animacao.seek(i)
            original = animacao.convert('RGBA').copy()  # Pillow compõe os quadros com disposal
            duracao = int(animacao.info.get('duration', 100))
            duracoes.append(max(10, duracao))
            recortado = _recortar(original, sessao, descontaminar, ajustes)
            quadros.append(_gif_paleta_transparente(recortado) if formato == 'gif' else recortado)
            if avisar:
                avisar(f'{entrada.name} • quadro {i + 1}/{quantidade}')
    if not quadros:
        raise ValueError('GIF vazio.')
    if formato == 'gif':
        quadros[0].save(saida, format='GIF', save_all=True,
                       append_images=quadros[1:], duration=duracoes, loop=loop,
                       transparency=0, disposal=2, optimize=False)
    else:
        quadros[0].save(saida, format='WEBP', save_all=True,
                       append_images=quadros[1:], duration=duracoes, loop=loop,
                       lossless=True, quality=100, method=4)
    return len(quadros)


def _processo_encoder(ffmpeg: str, entrada: Path, saida: Path, w: int, h: int, fps: float, log):
    # Vídeo 0 vem da sequência RGBA gerada pela IA; áudio 1 vem do vídeo original.
    # VP9 alpha usa yuva420p. MP4/H.264 NÃO suporta esse alfa.
    args = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgba', '-video_size', f'{w}x{h}',
            '-framerate', f'{fps:.8f}', '-i', 'pipe:0', '-i', str(entrada),
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p', '-auto-alt-ref', '0', '-b:v', '0',
            '-crf', '28', '-deadline', 'good', '-cpu-used', '5',
            '-c:a', 'libopus', '-b:a', '128k', '-shortest', str(saida)]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)


def processar_video(
    entrada: Path, saida: Path, sessao, descontaminar: bool, ajustes: dict,
    avisar: Callable[[str], None] | None = None,
) -> int:
    """Recorta por streaming; WebM/VP9 alfa, preservando a faixa de áudio se houver."""
    import imageio_ffmpeg
    entrada, saida = Path(entrada).resolve(), Path(saida).resolve()
    if entrada == saida:
        raise ValueError('A saída não pode sobrescrever o vídeo de entrada.')
    if saida.suffix.lower() != '.webm':
        raise ValueError('Para transparência real em vídeo, exporte para .webm (VP9 alfa).')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    leitor = imageio_ffmpeg.read_frames(str(entrada), pix_fmt='rgb24')
    encoder = None
    count = 0
    try:
        meta = next(leitor)
        w, h = meta['size']
        fps = float(meta.get('fps') or 0)
        if not w or not h or fps <= 0:
            raise RuntimeError('Não foi possível identificar resolução/FPS do vídeo.')
        estimativa = int(fps * float(meta.get('duration') or 0))
        with tempfile.TemporaryFile(mode='w+b') as log:
            encoder = _processo_encoder(ffmpeg, entrada, saida, w, h, fps, log)
            try:
                assert encoder.stdin is not None
                for buffer in leitor:
                    frame = Image.frombytes('RGB', (w, h), buffer).convert('RGBA')
                    recorte = _recortar(frame, sessao, descontaminar, ajustes)
                    encoder.stdin.write(recorte.tobytes())
                    count += 1
                    if avisar and (count == 1 or count % 5 == 0):
                        referencia = f'/{estimativa} (aprox.)' if estimativa else ''
                        avisar(f'{entrada.name} • quadro {count}{referencia}')
            except (BrokenPipeError, OSError) as exc:
                encoder.stdin.close()
                encoder.wait()
                log.seek(0)
                detalhe = log.read().decode('utf-8', errors='replace')[-2000:]
                raise RuntimeError(f'Falha na codificação WebM: {detalhe or exc}') from exc
            finally:
                if encoder.stdin and not encoder.stdin.closed:
                    encoder.stdin.close()
            code = encoder.wait()
            if code != 0:
                log.seek(0)
                raise RuntimeError(f'FFmpeg não conseguiu criar o vídeo: {log.read().decode("utf-8", errors="replace")[-2000:]}')
        if count == 0:
            raise RuntimeError('O vídeo não continha quadros decodificáveis.')
        return count
    finally:
        leitor.close()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def processar_midia(
    entrada: str | Path, saida: str | Path, sessao,
    descontaminar: bool, ajustes: dict,
    formato_gif: str = 'gif', avisar: Callable[[str], None] | None = None,
) -> int:
    entrada, saida = Path(entrada), Path(saida)
    tipo = tipo_arquivo(entrada)
    if tipo == 'gif':
        if saida.suffix.lower() != f'.{formato_gif}':
            raise ValueError('Extensão de saída da animação não corresponde ao formato selecionado.')
        return processar_gif(entrada, saida, sessao, descontaminar, ajustes, formato_gif, avisar)
    if tipo == 'video':
        return processar_video(entrada, saida, sessao, descontaminar, ajustes, avisar)
    from motor import imagem_ajustada
    if saida.suffix.lower() != '.png':
        raise ValueError('A saída de imagem estática deve ser PNG.')
    imagem_ajustada(entrada, saida, sessao, descontaminar, **ajustes)
    return 1
