"""Lote de imagens, GIFs e vídeos, com saída em pasta ou ZIP local."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable
from zipfile import ZipFile, ZIP_STORED

from motor import imagem_ajustada
from midia import processar_midia, tipo_arquivo


def _nome_livre(stem: str, usados: set[str], pasta: Path | None = None, ext: str = '.png') -> str:
    base = f'{stem}_sem_fundo'
    numero = 1
    while True:
        nome = f'{base}{f"_{numero}" if numero > 1 else ""}{ext}'
        if nome.casefold() not in usados and (pasta is None or not (pasta / nome).exists()):
            usados.add(nome.casefold())
            return nome
        numero += 1


def gerar_lote(
    arquivos: list[Path], destino: Path, sessao, descontaminar: bool,
    ajustes: dict, em_zip: bool,
    avisar: Callable[[str], None] | None = None,
    processador=imagem_ajustada,
    formato_gif: str = 'gif',
    processador_animacao=processar_midia,
) -> tuple[int, list[str]]:
    """Um arquivo por entrada. Falhas individuais não interrompem os demais."""
    destino = Path(destino)
    acertos = 0
    falhas: list[str] = []
    usados: set[str] = set()
    if formato_gif not in ('gif', 'webp'):
        raise ValueError('Formato de GIF inválido.')

    def processar(entrada: Path, saida: Path):
        tipo = tipo_arquivo(entrada)
        if tipo == 'imagem':
            processador(entrada, saida, sessao, descontaminar, **ajustes)
        else:
            processador_animacao(entrada, saida, sessao, descontaminar, ajustes,
                                formato_gif=formato_gif, avisar=avisar)

    def extensao(arquivo: Path):
        tipo = tipo_arquivo(arquivo)
        return '.png' if tipo == 'imagem' else ('.webm' if tipo == 'video' else f'.{formato_gif}')

    def executar(pasta_tmp: Path, pacote: ZipFile | None = None):
        nonlocal acertos
        for indice, arquivo in enumerate(arquivos, 1):
            arquivo = Path(arquivo)
            if avisar:
                avisar(f'Arquivo {indice}/{len(arquivos)}: {arquivo.name}')
            ext = extensao(arquivo)
            nome = _nome_livre(arquivo.stem, usados, None if pacote else destino, ext)
            saida_tmp = pasta_tmp / nome
            try:
                processar(arquivo, saida_tmp)
                if not saida_tmp.exists() or saida_tmp.stat().st_size == 0:
                    raise RuntimeError('O arquivo gerado está vazio.')
                if pacote:
                    # ZIP streaming: evita manter vídeos e GIFs grandes na memória.
                    pacote.write(saida_tmp, arcname=nome)
                else:
                    os.replace(saida_tmp, destino / nome)
                acertos += 1
            except Exception as exc:
                falhas.append(f'{arquivo.name}: {type(exc).__name__}: {exc}')
            finally:
                if saida_tmp.exists():
                    saida_tmp.unlink()

    if em_zip:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.fundozero_', dir=destino.parent) as trabalho:
            pasta_tmp = Path(trabalho)
            pacote_tmp = pasta_tmp / 'pacote.zip'
            with ZipFile(pacote_tmp, 'w', compression=ZIP_STORED, allowZip64=True) as z:
                executar(pasta_tmp, z)
                if falhas and acertos:
                    z.writestr('ERROS_DO_LOTE.txt', '\n'.join(falhas).encode('utf-8'))
            if acertos:
                os.replace(pacote_tmp, destino)
    else:
        destino.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.fundozero_', dir=destino) as trabalho:
            executar(Path(trabalho))
    return acertos, falhas
