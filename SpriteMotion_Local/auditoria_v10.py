"""Regression v10: guide rectangle never changes frame scale/placement.
No model download required. Uses a local deterministic mock segmentation network.
"""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import sprite_engine as se

source=Path(__file__).with_name('SpriteMotion.py').read_text(encoding='utf-8')
ast.parse(source)
assert 'Reprocessar máscaras sem recorte móvel' in source
assert 'reprocess_stable_existing' in source
assert 'stable_reprocess_button.setEnabled(False)' in source
assert 'stable_reprocess_button.setEnabled(True)' in source
code=Path(__file__).with_name('sprite_engine.py').read_text(encoding='utf-8')
ast.parse(code)
assert 'work = original.crop(' not in code
assert 'reduced = original.copy()' in code
assert 'signatures.append(significant_signature(frame))' in code
print('V10 sintaxe/integração GUI/recorte completo: OK')

# The stub emulates an alpha segmentation model: saturated red/blue artwork
# is opaque; plain background is transparent; received dimensions recorded.
calls=[]
def fake_remove(image, session=None, alpha_matting=False):
    calls.append(image.size)
    rgb=np.asarray(image.convert('RGB'))
    red=(rgb[:,:,0]>155) & (rgb[:,:,1]<105) & (rgb[:,:,2]<110)
    blue=(rgb[:,:,2]>150) & (rgb[:,:,0]<100)
    alpha=np.where(red|blue,255,0).astype('uint8')
    rgba=np.concatenate([rgb,alpha[:,:,None]],axis=2)
    return Image.fromarray(rgba,'RGBA')
old=sys.modules.get('rembg')
sys.modules['rembg']=SimpleNamespace(remove=fake_remove,new_session=lambda x:None)
try:
    with TemporaryDirectory() as td:
        d=Path(td)
        image=np.full((160,200,3),(48,110,83),dtype='uint8')
        # Main subject red body: x90-118 and separate interfering blue object.
        image[30:135,90:119]=[220,45,35]
        image[65:115,13:34]=[15,25,225]
        raw=d/'raw.png';Image.fromarray(image,'RGB').save(raw)
        result_paths=[]
        boxes=[(79,20,54,124),(81,22,54,124),(77,19,54,124),(83,23,54,124)]
        for k,roi in enumerate(boxes):
            output=d/'project'/'sprites_rgba'/f'sprite_{k:06}.png'
            se.remove_bg_file(str(raw),str(output),None,roi=roi,cleanup_mode='off')
            result_paths.append(output)
        assert calls==[(200,160)]*4, f'Dynamic model crops used: {calls}'
        arrays=[np.asarray(Image.open(p).convert('RGBA')) for p in result_paths]
        assert all(np.array_equal(arrays[0],a) for a in arrays[1:]), 'ROI jitter changed output!'
        alpha=arrays[0][:,:,3]
        assert alpha[85,100]>200, 'subject lost'
        assert alpha[85,20]==0, 'distant object not filtered'
        # Repositioning ROI by several px must not shift bbox in full canvas.
        for a in arrays:
            yy,xx=np.where(a[:,:,3]>127)
            assert (xx.min(),xx.max(),yy.min(),yy.max())==(90,118,30,134)
        # Without guide, the general model is still usable (both components).
        plain=d/'project'/'sprites_rgba'/'sprite_plain.png'
        se.remove_bg_file(str(raw),str(plain),None,roi=None,cleanup_mode='off')
        assert np.asarray(Image.open(plain))[85,20,3]>200
        print('V10 caixa vibrando, modelo recebe frame 200x160 fixo, personagem estável, cenário não-verde: OK')
        # Test preserving manual corrections via backups, and restoration on errors.
        g=se.Group('Teste',str(raw),12,str(d/'project'),frames=[
            se.Frame(i,i/12,str(raw),str(result_paths[i]),True,roi=boxes[i]) for i in range(4)])
        # Change 1 of the previously retouched sprites to transparent (edit).
        edited=arrays[0].copy();edited[55,100,3]=0
        Image.fromarray(edited,'RGBA').save(result_paths[0])
        b=se.reprocess_stable_existing(g,[0,1],'fake',cleanup_mode='off')
        assert Path(b).is_dir()
        assert np.asarray(Image.open(Path(b)/'sprites_rgba'/result_paths[0].name))[55,100,3]==0
        assert np.asarray(Image.open(result_paths[0]))[55,100,3]==255
        print('V10 reprocessamento com backup de retoques e recuperação de máscara estável: OK')
        # Simulate cancellation after the FIRST modified item, verify full rollback.
        before=[p.read_bytes() for p in result_paths]
        st={'completed':0}
        def progress(percent,msg):
            st['completed']+=1
        try:
            se.reprocess_stable_existing(g,[0,1],'fake',cleanup_mode='off',
                callback=progress,cancelled=lambda:st['completed']>=1)
        except InterruptedError:
            pass
        assert st['completed']==1, 'rollback case never processed any frame'
        assert all(p.read_bytes()==buf for p,buf in zip(result_paths,before))
        print('V10 cancelamento após primeiro PNG / rollback total: OK')
finally:
    if old is None:
        sys.modules.pop('rembg',None)
    else:
        sys.modules['rembg']=old
print('AUDITORIA V10: OK')
