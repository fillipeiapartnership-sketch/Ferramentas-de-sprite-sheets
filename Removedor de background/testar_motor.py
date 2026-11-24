"""Testes locais dos ajustes que não precisam baixar o modelo de IA."""
from PIL import Image
from motor import Traco, finalizar_imagem, construir_pinceis


def main():
    original = Image.new('RGBA', (101, 101), (200, 140, 90, 255))
    recorte = original.copy()
    alpha = Image.new('L', original.size, 0)
    from PIL import ImageDraw
    ImageDraw.Draw(alpha).rectangle((25, 25, 75, 75), fill=255)
    recorte.putalpha(alpha)

    teste = finalizar_imagem(original, recorte)
    assert teste.getpixel((0, 0))[3] == 0
    assert teste.getpixel((50, 50))[3] == 255
    assert teste.size == original.size

    contraido = finalizar_imagem(original, recorte, borda=-3)
    assert contraido.getpixel((25, 25))[3] == 0
    expandido = finalizar_imagem(original, recorte, borda=3)
    assert expandido.getpixel((23, 23))[3] == 255

    tracos = [Traco('restaurar', 8, ((10, 10),)), Traco('apagar', 8, ((50, 50),))]
    pintado = finalizar_imagem(original, recorte, tracos=tracos)
    assert pintado.getpixel((10, 10))[3] == 255
    assert pintado.getpixel((50, 50))[3] == 0
    tracos.append(Traco('restaurar', 8, ((50, 50),)))
    pintado = finalizar_imagem(original, recorte, tracos=tracos)
    assert pintado.getpixel((50, 50))[3] == 255

    # PNG transparente preserva alpha
    from io import BytesIO
    f = BytesIO()
    teste.save(f, format='PNG')
    f.seek(0)
    assert Image.open(f).getpixel((0, 0))[3] == 0
    print('OK: transparência, recorte, pincel, undo lógico e exportação PNG.')


if __name__ == '__main__':
    main()
