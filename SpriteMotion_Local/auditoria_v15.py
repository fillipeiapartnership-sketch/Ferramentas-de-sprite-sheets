"""V15: uniform dark background mask and no-regression smoke tests. Offline."""
import numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
import tempfile
from uniform_black import cut_uniform_dark, is_uniform_dark_background
from sprite_engine import remove_bg_file,geometry,render_cell

# Black background + black outlines must be retained, without rembg installed.
w,h=260,360
raw=Image.new('RGB',(w,h),(0,0,0))
d=ImageDraw.Draw(raw)
d.ellipse((45,25,210,295),fill=(105,30,211),outline=(5,3,8),width=6)
d.rectangle((95,180,155,330),fill=(209,31,29),outline=(5,3,8),width=6)
rgba=cut_uniform_dark(raw)
assert rgba is not None
arr=np.asarray(rgba)
assert arr[0,0,3]==0 and arr[10,10,3]==0
assert arr[130,130,3]==255
assert arr[130,45,3]>200, 'Black outline wrongly erased'
with tempfile.TemporaryDirectory() as tmp:
    rawfile=Path(tmp)/'original.png';raw.save(rawfile)
    output=Path(tmp)/'sprites_rgba'/'frame.png'
    remove_bg_file(str(rawfile),str(output),{'instance':None,'model':'isnet-anime'},cleanup_mode='auto')
    assert output.is_file()
    saved=np.asarray(Image.open(output))
    assert saved[0,0,3]==0 and saved[130,45,3]>200
    g=geometry([str(output)],160,200,False,True)
    sprite=render_cell(str(output),g)
    assert sprite.mode=='RGBA' and sprite.size==(160,200)
print('Fundo preto, contorno preto, transparência, geometria: OK')
# Nonuniform dark gradients should NOT take the simplistic uniform key route.
grad=np.zeros((200,250,3),np.uint8)
for row in range(200): grad[row,:,:]=[row//5,row//4,row//3]
grad=np.asarray(grad)
assert not is_uniform_dark_background(grad)
assert cut_uniform_dark(Image.fromarray(grad)) is None
# Green/blue backdrops must use existing network pipeline.
for bg in ((10,220,20),(32,43,145)):
    im=Image.new('RGB',(260,360),bg)
    ImageDraw.Draw(im).ellipse((40,30,210,320),fill=(180,30,130))
    assert cut_uniform_dark(im) is None
# All black input: no subject, so do not manufacture a sprite.
assert cut_uniform_dark(Image.new('RGB',(260,360),'black')) is None
print('Gradiente escuro, fundo verde/azul e frames vazios: fallback preservado')
print('AUDITORIA V15: APROVADA')
