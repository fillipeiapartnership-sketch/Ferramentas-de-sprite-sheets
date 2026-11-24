"""Testes de GIF/WebP/WebM, transparência, duração, áudio e ZIP, sem baixar IA."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from zipfile import ZipFile

import imageio_ffmpeg
from PIL import Image, ImageDraw

import motor
from midia import primeira_imagem, processar_midia, tipo_arquivo
from lote import gerar_lote


def modelo_fake(original, sessao, descontaminar):
    im = original.convert('RGBA')
    # Pixels escuros são transparentes, quadrado colorido é opaco.
    rgba = []
    for r, g, b, a in im.getdata():
        rgba.append((r, g, b, a if max(r, g, b) > 40 else 0))
    im.putdata(rgba)
    return im


def _video_entrada(destino: Path, audio: bool):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
           '-i', 'testsrc2=size=48x48:rate=5:duration=1']
    if audio:
        cmd += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-shortest', '-c:a', 'aac']
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(destino)]
    subprocess.run(cmd, check=True, capture_output=True)


def _alpha_video(path: Path):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    # Libvpx decoder must be explicitly requested, otherwise ffmpeg decoder can drop alpha.
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-c:v', 'libvpx-vp9', '-i', str(path),
           '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-vframes', '1', '-']
    result = subprocess.run(cmd, check=True, capture_output=True)
    return result.stdout


def main():
    with tempfile.TemporaryDirectory() as work:
        p = Path(work)
        frames = []
        for i in range(3):
            im = Image.new('RGB', (32, 26), (0, 0, 0))
            ImageDraw.Draw(im).rectangle((4 + i * 5, 6, 15 + i * 5, 17), fill=(240, 90, 30))
            frames.append(im)
        entrada = p / 'sprite.gif'
        frames[0].save(entrada, save_all=True, append_images=frames[1:],
                       duration=[80, 120, 160], loop=0, disposal=2)
        assert tipo_arquivo(entrada) == 'gif'
        assert primeira_imagem(entrada).size == (32, 26)
        ajustes = {'borda': 0, 'suavidade': 0, 'limpeza': 0}
        with patch.object(motor, 'remover_fundo', side_effect=modelo_fake):
            # midia imported remover_fundo into namespace; patch at call site.
            from unittest.mock import patch as patch1
            with patch1('midia.remover_fundo', side_effect=modelo_fake):
                gif = p / 'transparente.gif'
                webp = p / 'transparente.webp'
                assert processar_midia(entrada, gif, object(), True, ajustes) == 3
                assert processar_midia(entrada, webp, object(), True, ajustes, formato_gif='webp') == 3
                with Image.open(gif) as im:
                    assert im.n_frames == 3, im.n_frames
                    assert im.info.get('loop') == 0
                    for i in range(3):
                        im.seek(i)
                        print('gif', i, 'duration', im.info.get('duration'))
                        assert im.getpixel((0,0)) == 0 if i == 0 else im.convert('RGBA').getpixel((0,0))[3] == 0
                        assert im.convert('RGBA').getpixel((7 + 5 * i, 9))[3] == 255
                with Image.open(webp) as im:
                    assert im.n_frames == 3
                    for i in range(3):
                        im.seek(i)
                        assert im.convert('RGBA').getpixel((0,0))[3] == 0
                video_input = p / 'entrada.mp4'
                _video_entrada(video_input, True)
                assert primeira_imagem(video_input).size == (48, 48)
                saida = p / 'saida.webm'
                assert processar_midia(video_input, saida, object(), True, ajustes) == 5
                pix = _alpha_video(saida)
                assert len(pix) == 48 * 48 * 4, len(pix)
                assert any(pix[k] == 0 for k in range(3, len(pix), 4)), 'Não há pixel transparente'
                assert any(pix[k] == 255 for k in range(3, len(pix), 4))
                exe = imageio_ffmpeg.get_ffmpeg_exe()
                info = subprocess.run([exe, '-hide_banner', '-i', str(saida)], capture_output=True, text=True)
                assert 'Audio: opus' in info.stderr, info.stderr
                assert 'alpha_mode' in info.stderr, info.stderr
                zipfile = p / 'lote.zip'
                sucessos, falhas = gerar_lote([entrada,video_input], zipfile, object(), True, ajustes, True)
                assert sucessos == 2 and not falhas, falhas
                with ZipFile(zipfile) as z:
                    assert z.testzip() is None
                    assert z.namelist() == ['sprite_sem_fundo.gif', 'entrada_sem_fundo.webm']
                    assert len(z.read('entrada_sem_fundo.webm')) > 100
    print('OK: GIF quadros, WebP transparente, vídeo VP9 alfa com áudio, ZIP misto.')


if __name__ == '__main__':
    main()
