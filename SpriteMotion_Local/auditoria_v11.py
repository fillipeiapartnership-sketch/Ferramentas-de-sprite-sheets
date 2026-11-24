"""Cycle selection regression. Offline, requires no rembg or Qt GUI."""
from pathlib import Path
import ast
import numpy as np, cv2
import sprite_engine as e

src=Path(__file__).with_name('SpriteMotion.py').read_text(encoding='utf-8')
assert 'suggest_cycle_window_for_group' in src
ast.parse(src)

def frames_from_pattern(phases, background=0):
    out=[]
    for n,phase in enumerate(phases):
        img=np.zeros((80,80,3),np.uint8)
        img[:]=(20+background,30+background,40+background)
        x=15+int(round(12*np.sin(2*np.pi*phase/8)))
        y=15+int(round(5*np.cos(2*np.pi*phase/8)))
        cv2.rectangle(img,(x,y),(x+24,y+40),(230,45,35),-1)
        out.append(e.significant_signature(img))
    return out

basic=frames_from_pattern([k%8 for k in range(48)])
assert e.detect_repeat_period(basic)==8
start,p=e.best_cycle_window(basic,8)
assert p==8 and all(start<=k<start+p for k in range(start,start+p))
# Codec / brightness drift: approximate recurrence should not be rejected.
drift=[]
for i in range(72):
 phase=i%8
 img=np.zeros((80,80,3),np.uint8); img[:]=(i//12+20,i//12+30,i//12+40)
 x=15+int(round(12*np.sin(2*np.pi*phase/8)))
 y=15+int(round(5*np.cos(2*np.pi*phase/8)))
 cv2.rectangle(img,(x,y),(x+24,y+40),(230,45,35),-1)
 drift.append(e.significant_signature(img))
assert e.detect_repeat_period(drift)==8
rng=np.random.default_rng(71)
noise=[rng.integers(0,256,(96,96,3),dtype=np.uint8) for _ in range(60)]
assert e.detect_repeat_period(noise) is None
still=[np.full((96,96,3),50,np.uint8) for _ in range(40)]
assert e.detect_repeat_period(still) is None
print('AUDITORIA V11: loops 8 quadros, drift, movimento aleatório e cena estática: OK')
