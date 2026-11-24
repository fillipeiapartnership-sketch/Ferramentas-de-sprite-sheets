"""Testes de seleção/cancelamento/confirmação sem Qt e sem downloads."""
from pathlib import Path
import ast
import numpy as np
from smart_retouch import select_retouch, apply_retouch_selection

source=Path(__file__).with_name('SpriteMotion.py').read_text(encoding='utf-8')
ast.parse(source)
assert "class SpriteHoverPreview(QFrame):" in source
assert 'self.frame_list.viewport().installEventFilter(self)' in source
assert 'self.confirm_button.clicked.connect(self.canvas.confirm_selection)' in source
assert 'self.cancel_selection_button.clicked.connect(lambda:self.canvas.cancel_selection())' in source
assert 'self.pending_selection=selected' in source
assert 'self.cancel_selection(' in source
assert "self.hover_popup.show_frame" in source
print('Interface: lupa hover, duplo clique, seleção, confirmação e descarte conectados.')

# Cor verde contaminando borda, com resto da imagem preservado.
sprite=np.zeros((128,128,4),np.uint8)
sprite[25:100,28:98]=[90,35,165,255]
sprite[53:58,24:28]=[25,145,45,245]
untouched=sprite.copy()
sel,msg=select_retouch(sprite,25,55,tolerance=18,radius=16,mode='erase')
assert sel is not None and sel.action=='erase',(sel,msg)
assert sel.mask.sum()==sel.count
assert np.array_equal(sprite,untouched), 'Clique modificou a imagem antes da confirmação!'
assert untouched[55,25,3]==245, 'Entrada de teste inválida'
# "Descartar" apenas libera a seleção, portanto o sprite fica intacto.
sel=None
assert np.array_equal(sprite,untouched), 'Descartar seleção alterou pixels'
sel,msg=select_retouch(sprite,25,55,tolerance=18,radius=16,mode='erase')
out,description,n=apply_retouch_selection(sprite,sel)
assert n>0 and out[55,25,3]==0 and np.array_equal(out[50,70],untouched[50,70])
assert np.array_equal(sprite,untouched), 'Confirmação deve retornar cópia para desfazer'
print('Seleção da mancha: clique não apaga; descartar não apaga; confirmar altera só os pixels selecionados.')

# Cor dentro da roupa não abre buraco, mas muda RGB ao confirmar.
sprite=np.zeros((110,110,4),np.uint8)
sprite[20:95,20:95]=[35,70,200,255]
sprite[51:56,52:57]=[220,20,35,255]
original=sprite.copy()
sel,msg=select_retouch(sprite,54,53,15,22,'auto')
assert sel is not None and sel.action=='color'
assert np.array_equal(sprite,original)
out,_,n=apply_retouch_selection(sprite,sel)
assert n>0 and out[53,54,3]==255
assert out[53,54,2] > original[53,54,2]
assert np.array_equal(out[30,30],original[30,30])
print('Mancha interna: seleção prévia, correção de cor confirmada, alfa preservado.')

# Não selecionar uma superfície grande de roupa.
sprite=np.zeros((150,150,4),np.uint8)
sprite[10:140,10:140]=[120,55,180,255]
sel,msg=select_retouch(sprite,75,75,80,40,'erase')
assert sel is None, (sel,msg)
print('Proteção do personagem: seleção excessiva bloqueada.')
print('AUDITORIA V12: APROVADA.')
