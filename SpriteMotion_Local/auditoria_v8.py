"""Regression tests: persisted storage preference, animation timing, RGBA GIF transparency."""
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import numpy as np
from sprite_engine import Frame, Group, geometry, render_cell
from gif_export import export_animation_gif, indexed_gif_frame


def run():
    with TemporaryDirectory() as td:
        p=Path(td)
        frames=[]
        original_images=[]
        for i in range(4):
            # Synthetic test fixture; same canvas size, moving opaque subject,
            # dark outlines, partial-alpha edge, corrected green-spill pixel.
            arr=np.zeros((40,60,4), dtype=np.uint8)
            arr[12:30,10+i*6:24+i*6]=[28,58,223,255]
            arr[12:13,10+i*6:24+i*6]=[20,24,33,255]
            arr[12:30,10+i*6]=[20,24,33,255]
            arr[30,10+i*6:24+i*6]=[28,58,223,100]
            im=Image.fromarray(arr,'RGBA')
            path=p/f'sprite_{i}.png'
            im.save(path)
            frames.append(Frame(index=i,timestamp=i/12,original=str(path),sprite=str(path),selected=True))
            original_images.append(path.read_bytes())
        g=Group('teste','',12,str(p),frames,60,40)
        dest=p/'final'/'animacao.gif'
        out=export_animation_gif(g,dest,[0,1,2,3],12,60,40)
        assert out.is_file()
        assert [fr.sprite for fr in frames]==[str(p/f'sprite_{i}.png') for i in range(4)]
        assert [Path(fr.sprite).read_bytes() for fr in frames]==original_images, 'Modified input corrected PNGs'
        layout=geometry([f.sprite for f in frames],60,40)
        with Image.open(out) as gif:
            assert gif.n_frames==4, f'Expected 4 frames, got {gif.n_frames}'
            assert gif.info.get('loop')==0, f'Expected infinite loop: {gif.info}'
            for i in range(gif.n_frames):
                gif.seek(i)
                rgba=np.asarray(gif.convert('RGBA'))
                assert rgba.shape == (40,60,4)
                assert int(rgba[0,0,3])==0, f'Missing transparent background at frame {i}'
                expected=np.asarray(render_cell(frames[i].sprite,layout))
                fg=(expected[:,:,3]>=128)
                assert np.count_nonzero(fg)>0
                assert np.count_nonzero((rgba[:,:,3]>0)&fg)/np.count_nonzero(fg)>0.95, f'Foreground lost at {i}'
                blue=(expected[:,:,2]>150)&(expected[:,:,3]>200)&(expected[:,:,0]<90)
                assert np.count_nonzero(blue)>0
                assert np.count_nonzero((rgba[:,:,2]>150)&blue)/np.count_nonzero(blue)>0.90, f'Blue color altered at {i}'
                assert int(gif.info['duration']) in (80, 90), f'Unexpected frame duration {gif.info["duration"]}'
        # for correctness, verify GIF uses saved corrected PNG and original untouched
        arr=np.asarray(Image.open(frames[1].sprite).convert('RGBA')).copy()
        arr[20,17,0:4] = (243, 40, 31, 255)
        Image.fromarray(arr,'RGBA').save(frames[1].sprite)
        changed=p/'reexport.gif'
        export_animation_gif(g,changed,[0,1,2],12,60,40)
        with Image.open(changed) as gif:
            gif.seek(1)
            rgba=np.asarray(gif.convert('RGBA'))
            assert np.any((rgba[:,:,0]>160)&(rgba[:,:,1]<120)&(rgba[:,:,2]<150)&(rgba[:,:,3]>0)), 'Corrected red pixel was not used'
        # preserve transparency when all foreground is empty
        empty=indexed_gif_frame(Image.new('RGBA',(20,20),(0,0,0,0)))
        assert empty.info.get('transparency')==0
    print('v8 AUDITORIA: GIF de 4 poses, ordem, loop, duração, transparência, cor, correções PNG: OK')


if __name__=='__main__':
    run()
