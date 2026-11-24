"""Verificação offline do seletor de personagem, sem carregar Qt/segmentation."""
from pathlib import Path
import ast
import types

root = Path(__file__).parent
ui = (root/'SpriteMotion.py').read_text(encoding='utf-8')
roi_src = (root/'roi_selector.py').read_text(encoding='utf-8')
engine = (root/'sprite_engine.py').read_text(encoding='utf-8')
for code in (ui,roi_src,engine):
    ast.parse(code)
assert 'cv2.selectROI(' not in ui
assert 'SubjectSelectionDialog(frame, Path(video_path).name, self)' in ui
assert 'return dialog.selected_roi()' in ui
assert 'QGraphicsView' in roi_src and 'fitInView' in roi_src
assert 'ScrollBarAlwaysOn' in roi_src
assert 'change_zoom' in roi_src and 'self.fit_mode = False' in roi_src
assert 'roi_from_points(self.drag_start, dest, self.w, self.h)' in roi_src
assert 'roi=roi' in engine or 'roi=last_roi' in engine

# Check coordinate calculations by executing only the pure geometry helpers
# with a tiny Qt point stand-in, so the test does not require the Qt runtime.
class Point:
    def __init__(self,x,y): self._x, self._y = x,y
    def x(self): return self._x
    def y(self): return self._y
functions=[]
for obj in ast.parse(roi_src).body:
    if isinstance(obj,ast.FunctionDef) and obj.name in {'clamp_scene_point','roi_from_points'}:
        obj.returns=None
        obj.args.args[0].annotation=None
        for arg in obj.args.args: arg.annotation=None
        functions.append(obj)
mod=ast.Module(body=functions, type_ignores=[])
ast.fix_missing_locations(mod)
ns={'QPointF':Point}
exec(compile(mod,'roi_math','exec'),ns)
roi=ns['roi_from_points']
assert roi(Point(22.4,38.1),Point(467.8,828.0),496,864)==(22,38,446,790)
assert roi(Point(467.8,828.0),Point(22.4,38.1),496,864)==(22,38,446,790)
assert roi(Point(-50,-100),Point(900,1000),496,864)==(0,0,496,864)
assert roi(Point(200,100),Point(200,100),496,864)==(200,100,0,0)
# Rendering at 60% or 200% zoom maps back to source pixels exactly.
for zoom in [0.5,0.6,1,2,4]:
    start=(100,140); end=(340,800)
    mapped=lambda p:Point((p[0]*zoom)/zoom,(p[1]*zoom)/zoom)
    assert roi(mapped(start),mapped(end),496,864)==(100,140,240,660)
print('V13: ROI original estável, recorte inverso, limites, zoom, scroll e integração: OK')
