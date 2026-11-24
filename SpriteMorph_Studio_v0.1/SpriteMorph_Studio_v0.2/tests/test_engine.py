import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import SpriteProject


def sprite(h:int, dx=0, canvas=(140,180),color=(40,180,230,255)):
    im=Image.new('RGBA',canvas,(0,0,0,0))
    dr=ImageDraw.Draw(im)
    # leg/torso/arms/head, pixel art silhouette
    cx=canvas[0]//2+dx;y=canvas[1]-12
    k=h/100
    dr.ellipse((cx-10*k,y-h,cx+10*k,y-h+19*k),fill=color)
    dr.rectangle((cx-8*k,y-h+20*k,cx+8*k,y-h+70*k),fill=color)
    dr.rectangle((cx-15*k,y-h+22*k,cx-9*k,y-h+54*k),fill=color)
    dr.rectangle((cx+9*k,y-h+22*k,cx+15*k,y-h+54*k),fill=color)
    dr.rectangle((cx-8*k,y-h+70*k,cx-2*k,y),fill=color)
    dr.rectangle((cx+2*k,y-h+70*k,cx+8*k,y),fill=color)
    return im

def pngbytes(im):
    out=io.BytesIO();im.save(out,format='PNG');return out.getvalue()

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zip=Path(self.tmp.name)/'jogo.zip'
        self.out=Path(self.tmp.name)/'jogo_novo.zip'
        with zipfile.ZipFile(self.zip,'w') as z:
            z.writestr('gamescene/config.txt',b'KEEP UNCHANGED')
            for i in range(3):
                z.writestr(f'assets/characters/hero_idle_{i:03}.png',pngbytes(sprite(85,dx=i-1)))
                z.writestr(f'assets/characters/hero_run_{i:03}.png',pngbytes(sprite(118,dx=i-1)))
            for i in range(2):
                z.writestr(f'assets/characters/villain_idle_{i:03}.png',pngbytes(sprite(72,color=(200,60,80,255))))
            # Opaque backdrop not a sprite
            z.writestr('assets/background.png',pngbytes(Image.new('RGBA',(250,130),(20,20,50,255))))
            # Native transparent sprite strip
            sheet=Image.new('RGBA',(360,120),(0,0,0,0))
            for i in range(3):sheet.alpha_composite(sprite(65,canvas=(90,120)),(120*i,0))
            z.writestr('assets/sheet_hero_walk.png',pngbytes(sheet))
            # Animated GIF
            frames=[sprite(65,canvas=(110,140),color=(255,100,100,255)),sprite(65,dx=5,canvas=(110,140),color=(255,100,100,255))]
            out=io.BytesIO()
            frames[0].save(out,format='GIF',save_all=True,append_images=frames[1:],duration=[100,120],loop=0,disposal=2)
            z.writestr('assets/villain_idle.gif',out.getvalue())

    def test_scan_and_export(self):
        p=SpriteProject();stats=p.scan(str(self.zip))
        self.assertGreaterEqual(stats['sprites_analisados'],10)
        hero=[g for g in p.groups.values() if 'hero' in g.key and len(g.ids)>=6]
        self.assertTrue(hero,[(g.key,len(g.ids)) for g in p.groups.values()])
        g=hero[0]
        self.assertTrue(g.enabled)
        g.reference=next(id for id in g.ids if 'idle' in p.items[id].file)
        report=p.export(str(self.out),full_integrity=True)
        self.assertTrue(self.out.exists())
        self.assertGreater(report['arquivos_editados'],0,report)
        self.assertGreater(report['frames_ajustados'],0,report)
        with zipfile.ZipFile(self.zip) as a,zipfile.ZipFile(self.out) as b:
            self.assertEqual(a.namelist(),b.namelist())
            self.assertEqual(a.read('gamescene/config.txt'),b.read('gamescene/config.txt'))
            self.assertEqual(a.read('assets/background.png'),b.read('assets/background.png'))
            self.assertIsNone(b.testzip())
            with Image.open(io.BytesIO(b.read('assets/characters/hero_run_000.png'))) as im:
                self.assertEqual(im.size,(140,180))
                self.assertTrue(im.getbbox())
        self.assertTrue(Path(str(self.out).replace('.zip','_auditoria.json')).exists())

    def test_preview_and_reference(self):
        p=SpriteProject();p.scan(str(self.zip))
        g=next(g for g in p.groups.values() if 'hero' in g.key and len(g.ids)>=6)
        g.reference=next(id for id in g.ids if 'idle' in p.items[id].file)
        it=next(it for it in p.items.values() if 'hero_run_001' in it.file)
        src,changed,audit=p.preview(it.id)
        self.assertEqual(src.size,changed.size)
        self.assertTrue(audit['adjusted'],audit)
        self.assertLess(changed.getbbox()[3]-changed.getbbox()[1],src.getbbox()[3]-src.getbbox()[1])

    def test_sprite_strip_and_gif_preserved(self):
        p=SpriteProject();p.scan(str(self.zip))
        cells=[it for it in p.items.values() if it.file=='assets/sheet_hero_walk.png']
        self.assertEqual(len(cells),3)
        gif=[it for it in p.items.values() if it.file=='assets/villain_idle.gif']
        self.assertEqual(len(gif),2)
        p.export(str(self.out))
        with zipfile.ZipFile(self.out) as z:
            with Image.open(io.BytesIO(z.read('assets/villain_idle.gif'))) as im:
                self.assertEqual(im.n_frames,2)

    def test_original_cannot_be_overwritten(self):
        p=SpriteProject();p.scan(str(self.zip))
        with self.assertRaises(ValueError):p.export(str(self.zip))

    def test_group_move_and_reassign_reference(self):
        p=SpriteProject();p.scan(str(self.zip))
        it=next(it for it in p.items.values() if 'villain' in it.file)
        p.assign_to_group([it.id],'Novo Personagem')
        self.assertIn('Novo Personagem',p.groups)
        p.set_reference('Novo Personagem',it.id)
        self.assertEqual(p.groups['Novo Personagem'].reference,it.id)

if __name__=='__main__':unittest.main()
