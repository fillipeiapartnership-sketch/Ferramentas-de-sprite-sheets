"""Testes offline do lote, ZIP, nomes repetidos, canal alfa e falhas."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

import motor
from lote import gerar_lote


def recorte_simulado(original, sessao, descontaminar):
    resultado = original.copy()
    alpha = Image.new('L', resultado.size, 0)
    for x in range(4, resultado.width - 4):
        for y in range(4, resultado.height - 4):
            alpha.putpixel((x, y), 255)
    resultado.putalpha(alpha)
    return resultado


def criar_entradas(pasta: Path):
    arquivos = []
    for nome in ('foto.jpg', 'foto.png', 'Personagem.webp'):
        caminho = pasta / nome
        Image.new('RGB', (32, 30), (130, 70, 200)).save(caminho)
        arquivos.append(caminho)
    quebrado = pasta / 'quebrado.jpg'
    quebrado.write_bytes(b'arquivo corrompido')
    arquivos.append(quebrado)
    return arquivos


def main():
    with TemporaryDirectory() as temporario:
        raiz = Path(temporario)
        arquivos = criar_entradas(raiz)
        progresso = []
        zip_destino = raiz / 'concluido.zip'
        with patch.object(motor, 'remover_fundo', side_effect=recorte_simulado):
            acertos, falhas = gerar_lote(arquivos, zip_destino, object(), True,
                {'borda': 0, 'suavidade': 0, 'limpeza': 0}, True, progresso.append)
        assert acertos == 3 and len(falhas) == 1, (acertos, falhas)
        assert len(progresso) == 4, progresso
        assert zip_destino.is_file(), 'ZIP não criado'
        with ZipFile(zip_destino) as zipf:
            assert zipf.testzip() is None, 'ZIP corrompido'
            nomes = zipf.namelist()
            assert nomes == ['foto_sem_fundo.png', 'foto_sem_fundo_2.png',
                'Personagem_sem_fundo.png', 'ERROS_DO_LOTE.txt'], nomes
            for nome in nomes[:3]:
                img = Image.open(BytesIO(zipf.read(nome)))
                assert img.mode == 'RGBA', img.mode
                assert img.size == (32, 30)
                assert img.getpixel((0, 0))[3] == 0
                assert img.getpixel((16, 15))[3] == 255
            assert 'quebrado.jpg' in zipf.read('ERROS_DO_LOTE.txt').decode('utf-8')

        pasta_saida = raiz / 'saida'
        pasta_saida.mkdir()
        arquivo_anterior = pasta_saida / 'foto_sem_fundo.png'
        arquivo_anterior.write_bytes(b'nao substituir')
        with patch.object(motor, 'remover_fundo', side_effect=recorte_simulado):
            acertos, falhas = gerar_lote(arquivos, pasta_saida, object(), True,
                {'borda': 0, 'suavidade': 0, 'limpeza': 0}, False)
        assert acertos == 3 and len(falhas) == 1
        assert arquivo_anterior.read_bytes() == b'nao substituir'
        assert sorted(f.name for f in pasta_saida.iterdir()) == [
            'Personagem_sem_fundo.png', 'foto_sem_fundo.png',
            'foto_sem_fundo_2.png', 'foto_sem_fundo_3.png']
        assert Image.open(pasta_saida / 'foto_sem_fundo_2.png').getpixel((0, 0))[3] == 0

        # Nenhum arquivo processado: não cria pacote vazio nem substitui ZIP existente.
        sentinel = b'zip antigo preservado'
        zip_destino.write_bytes(sentinel)
        acertos, falhas = gerar_lote([arquivos[-1]], zip_destino, object(), False,
            {'borda': 0, 'suavidade': 0, 'limpeza': 0}, True)
        assert acertos == 0 and len(falhas) == 1
        assert zip_destino.read_bytes() == sentinel
        assert not list(raiz.glob('.fundozero_*')), 'Arquivos temporários sobrando'
    print('OK: lote em pasta, ZIP transparente, colisões, falhas e preservação de arquivos.')


if __name__ == '__main__':
    main()
