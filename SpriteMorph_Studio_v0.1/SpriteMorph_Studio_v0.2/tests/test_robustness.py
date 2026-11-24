import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image,ImageDraw
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import SpriteProject


def mk(h,subject_width=20):
    im=Image.new('RGBA',(120,160))
    ImageDraw.Draw(im).rectangle((60-subject_width//2,140-h,60+subject_width//2,140),fill=(230,80,100,255))
    data=io.BytesIO();im.save(data,'PNG');return data.getvalue()

class Robustness(unittest.TestCase):
    def test_isolated_outlier_is_corrected_without_distorting_other_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'in.zip';out=Path(tmp)/'out.zip'
            with zipfile.ZipFile(source,'w') as z:
                for i,height in enumerate([80,80,80,108,80]):
                    z.writestr(f'hero_idle_{i:03}.png',mk(height))
                for i in range(4):
                    z.writestr(f'hero_run_{i:03}.png',mk(100))
                z.writestr('config/settings.ini',b'preserved')
            p=SpriteProject();p.scan(str(source))
            group=next(iter(p.groups.values()))
            self.assertTrue(group.enabled)
            it=next(i for i in p.items.values() if 'idle_003' in i.file)
            src,mod,audit=p.preview(it.id)
            self.assertTrue(audit['adjusted'])
            self.assertLess(mod.getbbox()[3]-mod.getbbox()[1],src.getbbox()[3]-src.getbbox()[1])
            p.export(str(out),full_integrity=True)
            with zipfile.ZipFile(source) as x,zipfile.ZipFile(out) as y:
                self.assertEqual(x.read('config/settings.ini'),y.read('config/settings.ini'))
                self.assertIsNone(y.testzip())

    def test_invalid_png_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            zippath=Path(tmp)/'project.zip';out=Path(tmp)/'out.zip'
            with zipfile.ZipFile(zippath,'w') as z:
                z.writestr('hero_idle_001.png',mk(80))
                z.writestr('hero_idle_002.png',mk(80))
                z.writestr('damaged.png',b'not a real image')
            p=SpriteProject();p.scan(str(zippath))
            self.assertTrue(any('damaged.png' in x for x in p.warnings))
            result=p.export(str(out))
            with zipfile.ZipFile(out) as z:
                self.assertEqual(z.read('damaged.png'),b'not a real image')
            self.assertIn('imagens_com_erro',p.scan_stats)

    def test_frame_size_and_pose_preserved(self):
        p=SpriteProject()
        from engine import Sprite
        im=Image.open(io.BytesIO(mk(40)))
        s=Sprite(id='x',file='x.png',width=120,height=160,bbox=im.getbbox())
        bigger,audit=p.render(im,s,1.8)
        self.assertEqual(im.size,bigger.size)
        self.assertTrue(bigger.getbbox())
        self.assertLessEqual(bigger.getbbox()[2],im.width)
        self.assertLessEqual(bigger.getbbox()[3],im.height)

    def test_protected_action_is_not_forced_to_standing_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'in.zip'
            with zipfile.ZipFile(source,'w') as z:
                for a,h in [('idle',95),('crouch',42)]:
                    for i in range(3):z.writestr(f'hero_{a}_{i:03}.png',mk(h))
            p=SpriteProject();p.scan(str(source))
            f=p._compute_factors()
            crouch=[f[i.id] for i in p.items.values() if 'crouch' in i.file]
            self.assertEqual(len(set(round(x,5) for x in crouch)),1)  # identical scale through the pose
            self.assertTrue(all(x > 0 for x in crouch))
            one=next(i for i in p.items.values() if 'crouch' in i.file)
            old,new,audit=p.preview(one.id)
            self.assertLess(new.getbbox()[3]-new.getbbox()[1], p.target_height)  # crouch remains bent

if __name__=='__main__':unittest.main()
