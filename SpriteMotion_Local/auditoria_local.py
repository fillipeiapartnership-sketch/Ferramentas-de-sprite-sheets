"""Synthetic end-to-end audit that does not download model weights."""
import json
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sprite_engine import Frame, Group, process_video, export_group, render_cell, geometry, foreground_similar, load_project, reapply_cleanup, detect_repeat_period, significant_signature
from green_cleanup import clean_green_spill


def test():
    with tempfile.TemporaryDirectory(prefix='audit_sprite_') as temp:
        root=Path(temp)
        video=root/'caminhada.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),12,(128,128))
        assert writer.isOpened()
        for i in range(36):
            step=i%6
            img=np.full((128,128,3),(220,170,30),np.uint8)
            cv2.rectangle(img,(44,16),(84,79),(40,200,190),-1)
            cv2.line(img,(52,77),(30+step*4,110),(0,0,0),7)
            cv2.line(img,(76,77),(100-step*4,110),(0,0,0),7)
            writer.write(img)
        writer.release()
        original=sys.modules.get('rembg')
        fake=types.ModuleType('rembg')
        fake.new_session=lambda name:object()
        def remove(im,session=None,**kwargs):
            rgb=np.array(im.convert('RGB'))
            a=(rgb[:,:,0]<150).astype('uint8')*255
            return Image.fromarray(np.dstack((rgb,a)),'RGBA')
        fake.remove=remove
        sys.modules['rembg']=fake
        try:
            grp=process_video(str(video),str(root/'job'),'isnet-anime',2)
            assert len(grp.frames)==36, len(grp.frames)
            assert len(list((root/'job'/'frames_originais').glob('*.png')))==36
            chosen=[i for i,frame in enumerate(grp.frames) if frame.selected]
            print('Frames extraídos:',len(grp.frames))
            assert all(fr.original.endswith('.png') for fr in grp.frames)
            print('Poses retidas:',len(chosen))
            assert 4<=len(chosen)<=7, chosen
            paths=[grp.frames[i].sprite for i in chosen]
            lay=geometry(paths,128,128,True)
            rendered=[render_cell(path,lay) for path in paths]
            assert all(im.size==(128,128) for im in rendered)
            assert all(im.mode=='RGBA' for im in rendered)
            assert any(0<np.count_nonzero(np.asarray(im)[:,:,3])<128*128 for im in rendered)
            assert (root/'job'/'recortes_base_ia').is_dir()
            amount,pages=export_group(grp,str(root/'out'),chosen,128,128,3,0,True)
            assert amount==len(chosen), amount
            atlas=json.loads((root/'out'/'atlas.json').read_text())
            assert len(atlas['frames'])==len(chosen)
            assert atlas['cell_size']=={'width':128,'height':128}
            with Image.open(root/'out'/'sprite_sheet_001.png') as sheet:
                assert sheet.mode=='RGBA'
                assert sheet.getpixel((0,0))[3]==0
            # The v2 project can be re-opened and cleanup reapplied from caches.
            recovered=load_project(str(root/'job'/'projeto.json'))
            assert len(recovered.frames)==len(grp.frames)
            assert len([f for f in recovered.frames if f.selected])==len(chosen)
            rewritten=reapply_cleanup(recovered,chosen,'off',100,5)
            assert len(rewritten)==len(chosen)
            rewritten=reapply_cleanup(recovered,chosen,'auto',75,6)
            assert len(rewritten)==len(chosen)
            print('Recuperação de projeto / reprocessamento de PNGs: OK')
            print('Sprite sheets gerados:',pages)
            print('Atlas JSON / transparência / tamanhos: OK')
        finally:
            if original is not None:sys.modules['rembg']=original
            else:sys.modules.pop('rembg',None)
    # Synthetic green-screen artifact: mixed green line next to a purple body,
    # red clothing and an entirely valid green shoe preserved in the interior.
    W,H=160,140
    rgb=np.zeros((H,W,3),np.uint8)
    rgb[:]=[18,212,23]
    rgb[25:112,48:109]=[106,35,167]   # purple person
    rgb[45:66,54:102]=[207,45,60]     # red clothing
    rgb[94:108,56:90]=[19,165,32]     # intentionally green accessory
    alpha=np.zeros((H,W),np.uint8)
    alpha[25:112,46:111]=255
    for x in (46,47,109,110):
        rgb[26:111,x]=[74,147,93]       # contaminated edge
    base=Image.fromarray(np.dstack((rgb,alpha)),'RGBA')
    source=Image.fromarray(rgb,'RGB')
    out=np.asarray(clean_green_spill(base,source,'auto',90,8))
    inp=np.asarray(base)
    assert out[50,50,1] == inp[50,50,1], 'purple inner body changed'
    assert np.array_equal(out[98,70],inp[98,70]), 'legitimate green shoe changed'
    # Our edge mixture is only about 4 px from a solid interior.
    assert out[75,47,1] < inp[75,47,1] or out[75,47,3]<inp[75,47,3], 'green fringe unmodified'
    out_off=np.asarray(clean_green_spill(base,source,'off',90,8))
    assert np.array_equal(inp,out_off), 'off mode modified pixels'
    no_green=rgb.copy()
    no_green[:]=[35,45,120]
    safe=np.asarray(clean_green_spill(Image.fromarray(np.dstack((no_green,alpha)),'RGBA'),
                                      Image.fromarray(no_green,'RGB'),'auto',90,8))
    assert np.array_equal(safe[:,:,:3],no_green), 'false activation on a blue backdrop'
    # The original video frame coordinates must not shift when legs move in the
    # opposite direction. Regression test for the v2 bbox-recenter jitter bug.
    with tempfile.TemporaryDirectory(prefix='audit_stability_') as tmp:
        atlas_paths=[]
        for i in range(8):
            pose=np.zeros((150,180,4),dtype=np.uint8)
            # Torso remains fixed while swinging limbs alter the global bbox.
            pose[22:89,88:113]=[210,70,50,255]
            end=20 if i%2==0 else 148
            cv2.line(pose,(100,90),(end,134),(235,160,65,255),9)
            fp=Path(tmp)/f'pose_{i}.png'
            Image.fromarray(pose,'RGBA').save(fp)
            atlas_paths.append(str(fp))
        for body_mode in (False,True):
            layout=geometry(atlas_paths,192,192,body_mode)
            centers=[]
            for path in atlas_paths:
                cell=np.asarray(render_cell(path,layout))
                red=(cell[:,:,0]>180)&(cell[:,:,1]<90)&(cell[:,:,2]<90)&(cell[:,:,3]>200)
                ys,xs=np.where(red)
                assert xs.size>40
                centers.append(float(xs.mean()))
            assert max(centers)-min(centers)<=2, (body_mode,centers)
        print('Regressão tremor: eixo de tronco fixo com pés variáveis: OK')

    # Test small nearly pure screen-green splashes inside opaque purple hair.
    stained=rgb.copy()
    stained[39:46,76:82]=[19,210,24]
    output=np.asarray(clean_green_spill(Image.fromarray(np.dstack((stained,alpha)),'RGBA'),
                                        Image.fromarray(stained,'RGB'),'auto',92,12))
    assert output[42,79,1] < 190, 'isolated internal green stain survived'
    assert output[42,79,3] == 255, 'internal stain left a hole in the sprite'
    assert np.array_equal(output[98,70],np.array([19,165,32,255])), 'green accessory changed'
    sample_signatures=[]
    for k in range(36):
        image=np.zeros((120,120,3),np.uint8)
        v=k%6
        cv2.rectangle(image,(35,20),(76,70),(200,120,70),-1)
        cv2.line(image,(40,65),(22+v*5,108),(150,80,220),6)
        sample_signatures.append(significant_signature(image))
    assert detect_repeat_period(sample_signatures)==6, 'full motion cycle was not detected'
    print('Detector de repetição: 6 quadros por ciclo, sem mesclar ciclos: OK')
    print('Manchas verdes internas: removidas sem furar ou danificar sapato verde: OK')
    print('Bordas verdes / roupa e acessórios / modo desligado / auto detecção: OK')

    # V4: dimensions may be mixed between imported source resolutions.
    # Normalization happens once per source-resolution class, not per walking
    # pose, so natural head bob and motion must remain intact.
    with tempfile.TemporaryDirectory(prefix='audit_auto_grid_v4_') as temp:
        root=Path(temp)
        sprites=[]
        raw_shapes=[(160,200),(320,400),(160,200),(320,400)]
        expected_heights=[]
        for idx,(width,height) in enumerate(raw_shapes):
            mul=width/160.
            px=np.zeros((height,width,4),np.uint8)
            torso_x=int(80*mul)
            # Torso and head should remain stable even as the legs swing.
            cv2.rectangle(px,(round(68*mul),round(28*mul)),
                          (round(92*mul),round(110*mul)),(220,70,55,255),-1)
            cv2.line(px,(torso_x,round(106*mul)),
                     (round((35 if idx%2==0 else 118)*mul),round(155*mul)),
                     (140,95,215,255),max(1,round(6*mul)))
            cv2.line(px,(torso_x,round(106*mul)),
                     (round((118 if idx%2==0 else 35)*mul),round(155*mul)),
                     (140,95,215,255),max(1,round(6*mul)))
            path=root/f'sprite_{idx}.png'
            Image.fromarray(px,'RGBA').save(path)
            sprites.append(str(path))
        layout=geometry(sprites,192,256,False,True)
        assert layout['mode']=='mixed'
        assert layout['source_sizes']==2
        assert len(layout['group_scale'])==2
        cells=[np.asarray(render_cell(path,layout)) for path in sprites]
        assert all(cell.shape==(256,192,4) for cell in cells)
        # The same character depicted at 1x/2x should have matching
        # foreground sizes, and identical torso center positions.
        bounding_heights=[]
        torso_centers=[]
        for cell in cells:
            yy,xx=np.where(cell[:,:,3]>50)
            bounding_heights.append(yy.max()-yy.min()+1)
            yy,xx=np.where((cell[:,:,0]>190)&(cell[:,:,1]<90)&
                           (cell[:,:,2]<90)&(cell[:,:,3]>200))
            assert xx.size>100
            torso_centers.append((float(xx.mean()),float(yy.mean())))
        assert max(bounding_heights)-min(bounding_heights)<=2, bounding_heights
        assert max(pt[0] for pt in torso_centers)-min(pt[0] for pt in torso_centers)<=2, torso_centers
        assert max(pt[1] for pt in torso_centers)-min(pt[1] for pt in torso_centers)<=2, torso_centers
        for torso_mode in (False,True):
            g=geometry(sprites,192,256,torso_mode,True)
            assert all(np.asarray(render_cell(path,g))[:,:,3].max()==255 for path in sprites)
        # Automatic fitting toggles on/off without changing RGBA validity.
        layout_off=geometry(sprites,192,256,False,False)
        assert layout_off['auto_fit'] is False
        assert all(v==1 for v in layout_off['group_scale'].values())
        # Exports use exact same geometry as the animation preview.
        fr=[Frame(index=i,timestamp=i/12,original=sprites[i],sprite=sprites[i],selected=True)
            for i in range(len(sprites))]
        grp=Group('mix','',12,str(root),fr)
        count,pages=export_group(grp,str(root/'export'),list(range(len(sprites))),
                                 192,256,2,0,False,auto_fit=True)
        assert count==len(sprites) and pages==1
        meta=json.loads((root/'export'/'atlas.json').read_text(encoding='utf-8'))
        assert meta['auto_fit_proportional']
        assert meta['source_resolution_groups']==2
        for i,cell in enumerate(cells,1):
            exported=np.asarray(Image.open(root/'export'/'sprites'/f'frame_{i:04}.png'))
            assert np.array_equal(exported,cell), 'preview differs from export'
        print('V4: imagens 160x200/320x400, proporções, alinhamento e exportação: OK')

    # V4: separate natural poses with the SAME canvas size should *never*
    # gain an independent zoom just because a leg is raised/extended.
    with tempfile.TemporaryDirectory(prefix='audit_pose_scale_v4_') as temp:
        paths=[]
        for i in range(5):
            rgba=np.zeros((200,160,4),np.uint8)
            rgba[28:110,65:91]=[210,65,55,255]
            foot_y=145+(i%2)*20
            cv2.line(rgba,(80,105),(65,foot_y),(130,100,190,255),8)
            pth=Path(temp)/f'same_{i}.png'
            Image.fromarray(rgba,'RGBA').save(pth)
            paths.append(str(pth))
        geom=geometry(paths,192,256,False,True)
        assert geom['mode']=='fixed' and geom['source_sizes']==1
        assert geom['group_scale']['(160, 200)']==1
        # Same subject's head/torso stays in the same place; leg movement
        # continues naturally and is not forcibly scaled away.
        heads=[];feet=[]
        for path in paths:
            img=np.asarray(render_cell(path,geom))
            head_ys,head_xs=np.where((img[:,:,0]>180)&(img[:,:,1]<90)&(img[:,:,3]>200))
            _,all_x=np.where(img[:,:,3]>200)
            heads.append((round(float(head_xs.mean()),2),round(float(head_ys.mean()),2)))
            yy=np.where((img[:,:,2]>160)&(img[:,:,3]>200))[0]
            feet.append(int(yy.max()))
        assert max(a[0] for a in heads)-min(a[0] for a in heads)<.1,heads
        assert max(a[1] for a in heads)-min(a[1] for a in heads)<.1,heads
        assert len(set(feet))==2,feet
        print('V4: caminhada com elevação de pernas não cria zoom/tremor: OK')

    print('AUDITORIA DE MOTOR: APROVADA')


if __name__=='__main__':test()
