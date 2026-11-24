"""Regression coverage for SpriteMorph 0.2 physical scale and atlas splitting."""
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import SpriteProject


def png(im):
    b=io.BytesIO();im.save(b,'PNG');return b.getvalue()


def actor(height,canvas=(140,180),left=50):
    image=Image.new('RGBA',canvas)
    d=ImageDraw.Draw(image)
    d.ellipse((left+15,canvas[1]-height,left+35,canvas[1]-height+20),fill=(250,120,40,255))
    d.rectangle((left+9,canvas[1]-height+22,left+38,canvas[1]-9),fill=(250,120,40,255))
    return image

class Improvements(unittest.TestCase):
    def test_walk_character_prefix_and_character_action_suffix(self):
        k1,t1=SpriteProject._group_track('src/assets/walk_blue_07.png')
        k2,t2=SpriteProject._group_track('src/assets/blue_block.png')
        k3,t3=SpriteProject._group_track('src/assets/blue_getup.png')
        self.assertEqual(k1,k2)
        self.assertEqual(k2,k3)
        self.assertEqual({t1.rsplit('::',1)[1],t2.rsplit('::',1)[1],t3.rsplit('::',1)[1]}, {'walk','block','getup'})

    def test_explicit_atlas_frames_survive_even_if_alpha_runs_not_regular(self):
        with tempfile.TemporaryDirectory() as tmp:
            z=Path(tmp)/'in.zip';out=Path(tmp)/'out.zip'
            sheet=Image.new('RGBA',(360,140))
            for i in range(3):
                im=actor(100,canvas=(120,140),left=36)
                sheet.alpha_composite(im,(120*i,0))
            with zipfile.ZipFile(z,'w') as f:
                f.writestr('src/game.js', 'const SHEETS={hero:{punch:["src/assets/hero_punch.png",3]}};')
                f.writestr('src/assets/hero_punch.png',png(sheet))
                for i in range(4):f.writestr(f'src/assets/walk_hero_{i:02}.png',png(actor(60)))
                f.writestr('src/assets/bg_station.png',png(Image.new('RGBA',(500,200),(5,8,12,255))))
            p=SpriteProject();st=p.scan(str(z))
            self.assertEqual(st['atlas_com_metadados'],1)
            self.assertEqual(len([it for it in p.items.values() if it.file.endswith('hero_punch.png')]),3)
            self.assertEqual(st['personagens_ativos'],1)
            p.canvas_expand=True;p.export(str(out),full_integrity=True)
            with zipfile.ZipFile(out) as f:
                self.assertEqual(f.read('src/game.js'),b'const SHEETS={hero:{punch:["src/assets/hero_punch.png",3]}};')
                self.assertEqual(f.read('src/assets/bg_station.png'),png(Image.new('RGBA',(500,200),(5,8,12,255))))
                edited=Image.open(io.BytesIO(f.read('src/assets/hero_punch.png')))
                self.assertEqual(edited.width % 3,0)
                self.assertEqual(edited.getbbox() is not None,True)

    def test_expand_single_sprite_preserves_pose_and_prevents_cropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            z=Path(tmp)/'in.zip';out=Path(tmp)/'out.zip'
            with zipfile.ZipFile(z,'w') as f:
                f.writestr('hero_idle.png',png(actor(60,canvas=(80,90),left=18)))
                for i in range(3):f.writestr(f'walk_hero_{i:02}.png',png(actor(110,canvas=(140,180))))
            p=SpriteProject();p.scan(str(z));p.canvas_expand=True
            ref=next(it for it in p.items.values() if it.file=='hero_idle.png')
            before,after,audit=p.preview(ref.id)
            self.assertTrue(audit['adjusted'])
            self.assertGreaterEqual(after.width,before.width)
            self.assertGreater(after.height,before.height)
            b=before.getbbox();a=after.getbbox()
            ratio_before=(b[2]-b[0])/(b[3]-b[1]);ratio_after=(a[2]-a[0])/(a[3]-a[1])
            self.assertAlmostEqual(ratio_before,ratio_after,delta=.06)
            p.export(str(out))
            with zipfile.ZipFile(out) as f:
                modified=Image.open(io.BytesIO(f.read('hero_idle.png')))
                self.assertEqual(modified.size,after.size)
                self.assertIsNone(f.testzip())

    def test_horizontal_getup_is_never_excluded_because_of_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            z=Path(tmp)/'in.zip'
            with zipfile.ZipFile(z,'w') as f:
                for i in range(4):f.writestr(f'walk_punk_{i:02}.png',png(actor(100)))
                for i in range(2):
                    frame=Image.new('RGBA',(420,130))
                    ImageDraw.Draw(frame).rectangle((25,60,400,105),fill=(200,80,170,255))
                    f.writestr(f'punk_getup_{i:02}.png',png(frame))
            p=SpriteProject();p.scan(str(z))
            self.assertTrue(all(it.eligible for it in p.items.values() if 'getup' in it.file))

if __name__=='__main__':unittest.main()
