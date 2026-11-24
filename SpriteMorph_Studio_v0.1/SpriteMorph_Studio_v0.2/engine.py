"""SpriteMorph Studio 0.2: silhouette-aware, game-asset-aware 2D sprite normalization.

No model weights or network calls. Group detection is heuristic and needs review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
import copy
import json
import math
import re
import shutil
import statistics
import threading
import time
import zipfile
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

SUPPORTED = {'.png', '.webp', '.gif', '.bmp'}
ACTION_WORDS = set('idle stand standing walk walking run running sprint dash attack attacking punch punching kick kicking '
                   'hurt hit damaged damage death dead die dying jump jumping air airborne fall falling crouch crouching '
                   'duck ducking roll rolling block blocking guard guarding climb climbing swim swimming shoot shooting '
                   'cast casting throw throwing slash slashing combo move movement back front side left right down up '
                   'frente costas lado parado andando correndo soco socando chute chutando pulando agachado agachar '
                   'dano ferido morto morte ataque defesa atacar andar correr andar'.split())
GENERIC_WORDS = set('assets asset sprites sprite sheets sheet images image imgs img graphics textures texture '
                    'characters character personagens personagem animations animation anim frames frame png raw '
                    'export exports data game gameassets resources res output output spritesheets '.split())
NEUTRAL_WORDS = set('idle stand standing walk walking run running parado andando correndo'.split())
PROTECTED_WORDS = set('crouch crouching duck ducking roll rolling death dead die dying fall falling jump jumping '
                      'air airborne prone lay lying slide sliding sleep sleeping sit sitting stagger '
                      'agachar agachado morto morte pulando caindo deitado sentado rolando'.split())
PERSON_WORDS = set('hero player enemy npc humanoid human zombie person character knight warrior mage rogue '
                   'girl boy man woman boss avatar soldier fighter ninja samurai pirate player1 protagonist '
                   'heroi heroina jogador inimigo personagem guerreiro soldado monstro garota menino garota'.split())
# Pillow 10 and 11 compatibility
RESAMPLE_NEAREST = Image.Resampling.NEAREST
RESAMPLE_SMOOTH = Image.Resampling.LANCZOS

MOTION_ALIASES = {
    "stand":"idle","standing":"idle","parado":"idle",
    "walking":"walk","andando":"walk","andar":"walk",
    "running":"run","correndo":"run","correr":"run",
    "punching":"punch","soco":"punch","socando":"punch",
    "kick":"kick","kicking":"kick","chute":"kick","chutando":"kick",
    "jumping":"jump","pulando":"jump","hit":"hit","hurt":"hit","damaged":"hit","dano":"hit",
    "falling":"fall","caindo":"fall","getup":"getup","levantar":"getup",
    "block":"block","blocking":"block","defesa":"block","guard":"block",
    "stun":"stun","stunned":"stun","uppercut":"upper","upper":"upper",
    "dive":"dive","voadora":"dive","crouching":"crouch","agachado":"crouch",
    "death":"death","dead":"death","morte":"death","morto":"death",
}
ACTION_WORDS.update({'getup','upper','uppercut','stun','stunned','dive','voadora','kneel','ajoelhado'})
MOTION_NAMES = set(ACTION_WORDS)|set(MOTION_ALIASES)|{"upper","uppercut","getup","stun","stunned","dive","voadora","kneel","ajoelhado","slide","roll"}
STANDING_MOTIONS = {"idle","walk","run","punch","hit","block"}
POSE_MOTIONS = {"fall","getup","stun","crouch","death","dive","jump","kick","upper","roll","slide"}



def _tokenize(x: str) -> list[str]:
    x = re.sub(r'([a-z])([A-Z])', r'\1_\2', x)
    tokens=[s for s in re.split(r'[^a-z0-9áéíóúâêôãõç]+', x.lower()) if s]
    # idle01 and run4 are actions with frame indices; enemy01 remains one identity token.
    expanded=[]
    for tok in tokens:
        match=re.fullmatch(r'([a-z]+)([0-9]+)',tok)
        if match and match.group(1) in ACTION_WORDS:
            expanded.extend(match.groups())
        else:
            expanded.append(tok)
    return expanded


def _tokens_for_path(name: str) -> list[str]:
    return _tokenize(PurePosixPath(name).stem)


def _bbox_from_mask(a: np.ndarray) -> Optional[tuple[int,int,int,int]]:
    if a.size == 0 or not a.any():
        return None
    rows = np.flatnonzero(a.any(axis=1))
    cols = np.flatnonzero(a.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1])+1, int(rows[-1])+1


def _runs(vals: np.ndarray) -> list[tuple[int,int]]:
    if not np.any(vals):
        return []
    x = np.flatnonzero(np.diff(np.r_[False, vals, False].astype(np.int8)))
    return [(int(x[i]), int(x[i+1])) for i in range(0,len(x),2)]


def _candidate_cells(img: Image.Image, name: str) -> list[tuple[int,int,int,int]]:
    """Only split actual transparent, regularly separated atlas cells; never crop their positions."""
    if img.width > 14000 or img.height > 14000:
        return [(0,0,img.width,img.height)]
    a = np.asarray(img.getchannel('A')) > 8
    if not a.any():
        return [(0,0,img.width,img.height)]
    cols = _runs(a.any(axis=0))
    rows = _runs(a.any(axis=1))
    name_hint = any(t in _tokens_for_path(name) for t in ('sheet','strip','atlas','spritesheet'))
    if len(cols) >= 3 and len(rows) <= 2:
        widths=[r-l for l,r in cols]
        if min(widths) >= 3 and max(widths) <= 3.2 * max(1,statistics.median(widths)):
            return [(l,0,r,img.height) for l,r in cols]
    if len(rows) >= 3 and len(cols) <= 2:
        heights=[b-t for t,b in rows]
        if min(heights) >= 3 and max(heights) <= 3.2 * max(1,statistics.median(heights)):
            return [(0,t,img.width,b) for t,b in rows]
    if len(cols)>=2 and len(rows)>=2 and len(cols)*len(rows) >= 4:
        widths=[r-l for l,r in cols]
        heights=[b-t for t,b in rows]
        if (max(widths) <= 2.3*max(1,min(widths)) and max(heights) <= 2.3*max(1,min(heights))):
            cells=[(l,t,r,b) for t,b in rows for l,r in cols if a[t:b,l:r].any()]
            if len(cells)>=3:
                return cells
    if name_hint and len(cols)>=2 and len(rows)==1:
        widths=[r-l for l,r in cols]
        if min(widths)>=3 and max(widths)<=4*min(widths):
            return [(l,0,r,img.height) for l,r in cols]
    return [(0,0,img.width,img.height)]


def _alpha_stats(img: Image.Image) -> tuple[Optional[tuple[int,int,int,int]],float,float]:
    small=img
    if max(img.size)>2000:
        small=img.copy()
        small.thumbnail((2000,2000),RESAMPLE_NEAREST)
    a=np.asarray(small.getchannel('A'))
    bbox=_bbox_from_mask(a>8)
    frac=float(np.mean(a>8))
    transparent=float(np.mean(a<248))
    if bbox is not None and small.size!=img.size:
        sx=img.width/small.width; sy=img.height/small.height
        bbox=tuple((max(0, math.floor(bbox[0]*sx)),max(0,math.floor(bbox[1]*sy)),
                   min(img.width,math.ceil(bbox[2]*sx)),min(img.height,math.ceil(bbox[3]*sy))))
    return bbox,frac,transparent


def _signature(img: Image.Image, bbox) -> tuple[float,...]:
    if not bbox:
        return (0.,)*12
    cut=img.crop(bbox)
    cut.thumbnail((40,40),RESAMPLE_NEAREST)
    a=np.array(cut.convert('RGBA'),dtype=np.uint8)
    valid=a[:,:,3]>100
    if not valid.any():
        return (0.,)*12
    rgb=a[:,:,:3][valid]
    # Weighted RGB distribution and shape: for human review / similarity only
    means=np.mean(rgb,axis=0)/255
    bins=[]
    for channel in range(3):
        h=np.histogram(rgb[:,channel],bins=3,range=(0,256))[0]
        bins.extend((h/max(1,len(rgb))).tolist())
    return tuple(round(float(v),3) for v in [*means,*bins])


@dataclass
class Sprite:
    id: str
    file: str
    frame: int = 0
    cell: tuple[int,int,int,int] | None = None
    width: int = 0
    height: int = 0
    bbox: tuple[int,int,int,int] | None = None
    coverage: float = 0.
    transparent: float = 0.
    signature: tuple[float,...] = ()
    group: str = ''
    track: str = ''
    score: float = 0.
    eligible: bool = False
    reason: str = ''
    format: str = ''
    scale: float = 1.
    changed: bool = False
    audit: list[str] = field(default_factory=list)
    ink_area: int = 0
    core_width: float = 0.0
    category: str = "sem_acao"
    atlas_fixed: bool = False

    @property
    def height_of_subject(self):
        return self.bbox[3]-self.bbox[1] if self.bbox else 0

    @property
    def width_of_subject(self):
        return self.bbox[2]-self.bbox[0] if self.bbox else 0


@dataclass
class Group:
    key: str
    ids: list[str] = field(default_factory=list)
    reference: str = ''
    enabled: bool = False
    confidence: str = 'baixa'


class Cancelled(Exception):
    pass


class SpriteProject:
    def __init__(self):
        self.input_path=''
        self.items: dict[str,Sprite] = {}
        self.groups: dict[str,Group] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.scan_stats: dict = {}
        self.cancel=threading.Event()
        self.last_export=''
        self.audit_report={}
        self.atlas_counts: dict[str,int] = {}
        self.target_height: float = 0.0
        self.canvas_expand: bool = False
        self.group_baselines: dict[str,float] = {}
        self.game_adapter = ""
        self.adapt_engine = True

    def _check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled('Operação cancelada pelo usuário')

    @staticmethod
    def _group_track(name: str) -> tuple[str,str]:
        """Resolve character + motion in either prefix order; frame numbers never identify a character."""
        p=PurePosixPath(name)
        stem=list(_tokens_for_path(name))
        folders=list(p.parts[:-1])
        directory=[]
        for part in reversed(folders):
            t=[v for v in _tokenize(part) if v not in GENERIC_WORDS and v not in MOTION_NAMES and not v.isdigit()]
            if t:
                directory=t
                break
        movements=[MOTION_ALIASES.get(t,t) for t in stem if t in MOTION_NAMES]
        if not movements:
            for part in reversed(folders):
                movements=[MOTION_ALIASES.get(t,t) for t in _tokenize(part) if t in MOTION_NAMES]
                if movements:break
        action=movements[0] if movements else 'sem_acao'
        identity=[t for t in stem if t not in MOTION_NAMES and t not in GENERIC_WORDS and not t.isdigit()]
        # 'enemy01' is an identity, unlike 'walk_03'.
        if not identity:
            identity=directory
        if not identity:
            identity=['nao_identificado']
        # Avoid conflating unrelated siblings from distinct scene folders when they have the same identity.
        scope=[q.lower() for q in folders if any(v not in GENERIC_WORDS and v not in MOTION_NAMES for v in _tokenize(q))]
        ident='_'.join(identity[:3])
        if scope and ident not in ('_'.join(directory),'nao_identificado'):
            key='/'.join(scope[-1:])+'/'+ident
        else:
            key=ident
        return key,key+'::'+action

    @staticmethod
    def _read_atlas_metadata(z: zipfile.ZipFile) -> dict[str,int]:
        """Identify explicitly declared equal-column strips, not hand/body gaps.

        Metro Fighters uses SHEETS character/motion tables with [path, frameCount].
        Other HTML/JS games can use the same [path, integer] convention.
        """
        found={}
        for info in z.infolist():
            if info.file_size > 5_000_000 or info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in ('.js','.html','.json'):
                continue
            try:
                text=z.read(info).decode('utf-8','replace')
            except Exception:
                continue
            for image,count in re.findall(r'[\"\']([^\"\']+?\.(?:png|webp|bmp))[\"\']\s*,\s*(\d{1,3})\s*\]',text,re.I):
                count=int(count)
                if not (2<=count<=120):continue
                image=image.replace('\\','/').lstrip('./')
                if image not in found:found[image]=count
        return found

    @staticmethod
    def _ink_metrics(img: Image.Image, bbox:tuple[int,int,int,int]|None) -> tuple[int,float]:
        if bbox is None:return 0,0.
        a=np.asarray(img.getchannel('A').crop(bbox))
        mask=a>20
        ink=int(mask.sum())
        if not ink:return 0,0.
        h=mask.shape[0]
        # Character body/core width: median width of occupied scanlines, excluding open arms.
        rows=np.sum(mask,axis=1)
        rows=rows[int(.18*h):max(int(.82*h),int(.18*h)+1)]
        core=float(np.percentile(rows[rows>0],60)) if (rows>0).any() else 0.
        return ink,core

    @staticmethod
    def _classify(img: Image.Image, bbox, coverage:float, transparent:float, name:str):
        if not bbox:
            return 0.,False,'Imagem sem pixels visíveis'
        x0,y0,x1,y1=bbox; w=x1-x0;h=y1-y0
        if w<6 or h<6:
            return 0.,False,'Imagem muito pequena'
        if transparent<0.008:
            return 0.,False,'Imagem opaca ou sem transparência; preservada'
        if coverage>0.91:
            return 0.,False,'Conteúdo ocupa quase toda a tela; possivelmente cenário'
        ratio=w/max(1,h)
        tokens=set(_tokenize(name))
        name_person=bool(tokens&PERSON_WORDS)
        name_backdrop=bool(tokens&{'background','bg','backdrop','tile','tiles','ground','floor','sky','cloud','terrain','parallax','cenario','fundo','chao','font'})
        if name_backdrop:
            return 0.,False,'Indicação de cenário ou textura no nome'
        score=0.25
        if 0.2<=ratio<=1.15: score+=0.28
        elif 0.12<=ratio<=2.0: score+=0.10
        if coverage<0.75:score+=0.12
        if name_person:score+=0.25
        if tokens&ACTION_WORDS:score+=0.2
        if 12<=w<=2000 and 20<=h<=2000:score+=0.07
        if name_person and tokens&ACTION_WORDS:score+=0.12
        score=min(1.,score)
        return score, score>=0.50,'Candidato humanoide (heurística; revisar visualmente)'

    def scan(self, input_path: str, notify:Optional[Callable[[str,int,int],None]]=None):
        self.cancel.clear(); self.input_path=str(input_path)
        self.items.clear(); self.groups.clear(); self.warnings.clear(); self.errors.clear()
        counts=Counter(); t0=time.time()
        with zipfile.ZipFile(self.input_path,'r',allowZip64=True) as z:
            self.atlas_counts=self._read_atlas_metadata(z)
            counts['atlas_com_metadados']=len(self.atlas_counts)
            self.game_adapter=''
            if 'src/ff_game.js' in z.namelist() and len(self.atlas_counts)>=15:
                script=z.read('src/ff_game.js').decode('utf-8','replace')
                if all(marker in script for marker in ['const REFS=', 'const TARGET_H=', 'function sliceFrames(',
                                                      'function charScale(', 'const SHEETS=', 'async function loadAll()', '// faixa de profundidade:']):
                    self.game_adapter='metro_fighters'
            counts['motor_do_jogo_reconhecido']=self.game_adapter or 'generico'

            entries=z.infolist()
            counts['arquivos_total']=len(entries)
            sprites=[e for e in entries if not e.is_dir() and PurePosixPath(e.filename).suffix.lower() in SUPPORTED]
            counts['imagens_encontradas']=len(sprites)
            file_names=Counter(e.filename for e in entries)
            if any(v>1 for v in file_names.values()):
                self.warnings.append('ZIP contém nomes de arquivos duplicados. Confira o resultado antes de usar no jogo.')
            for ix,info in enumerate(sprites,1):
                self._check_cancel()
                try:
                    with z.open(info) as source:
                        with Image.open(source) as im:
                            fmt=im.format or PurePosixPath(info.filename).suffix[1:].upper()
                            fr_count=getattr(im,'n_frames',1)
                            if fr_count>1 and fmt=='GIF':
                                # Review frames separately, but not split sheet inside animated GIF
                                frames=range(fr_count)
                            else:
                                frames=range(1)
                            for fi in frames:
                                if fi:im.seek(fi)
                                rgba=im.convert('RGBA')
                                # GIF frames can be coalesced by Pillow; this preserves visual frame
                                declared=self.atlas_counts.get(info.filename.replace('\\','/')) if fi==0 and fr_count==1 else None
                                if declared:
                                    cells=[(round(i*rgba.width/declared),0,round((i+1)*rgba.width/declared),rgba.height) for i in range(declared)]
                                else:
                                    cells=_candidate_cells(rgba,info.filename) if fi==0 and fr_count==1 else [(0,0,rgba.width,rgba.height)]
                                for ci,cell in enumerate(cells):
                                    sub=rgba.crop(cell)
                                    bbox,coverage,trans=_alpha_stats(sub)
                                    score,eligible,reason=self._classify(sub,bbox,coverage,trans,info.filename)
                                    group,track=self._group_track(info.filename)
                                    id_=f'{info.filename}#f{fi}#c{ci}'
                                    ink,core=self._ink_metrics(sub,bbox)
                                    obj=Sprite(id=id_,file=info.filename,frame=fi,cell=cell if len(cells)>1 else None,
                                               width=sub.width,height=sub.height,bbox=bbox,coverage=coverage,
                                               transparent=trans,signature=_signature(sub,bbox),group=group,track=track,
                                               score=score,eligible=eligible,reason=reason,format=fmt,
                                               ink_area=ink,core_width=core,category=track.rsplit("::",1)[-1],atlas_fixed=bool(declared))
                                    self.items[id_]=obj
                                    counts['sprites_analisados']+=1
                                    if eligible:counts['candidatos']+=1
                                rgba.close()
                except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,MemoryError,EOFError,RuntimeError,zipfile.BadZipFile) as ex:
                    self.warnings.append(f'Não foi possível analisar {info.filename}: {str(ex)[:160]}')
                    counts['imagens_com_erro']+=1
                if notify and (ix%5==0 or ix==len(sprites)):
                    notify('Analisando sprites',ix,len(sprites))
        for it in self.items.values():
            # Keep uncertain sprites visible for manual review, but inactive by default.
            g=self.groups.setdefault(it.group,Group(key=it.group))
            g.ids.append(it.id)
        for g in self.groups.values():
            g.ids.sort(key=lambda sid:(self.items[sid].file,self.items[sid].frame,self.items[sid].cell or (0,0,0,0)))
            g.reference=self._pick_reference(g)
            confidence=self._group_confidence(g)
            g.confidence=confidence
            # Groups with only one frame do not need normalization
            g.enabled=((len(g.ids)>=2 and
                       sum(self.items[sid].eligible for sid in g.ids)/max(1,len(g.ids))>=0.60 and
                       (confidence!='baixa' or len(g.ids)>=6))
                       or (len(g.ids)==1 and self.items[g.ids[0]].eligible and
                           self.items[g.ids[0]].score>=.88 and
                           bool(set(_tokenize(self.items[g.ids[0]].file)) & PERSON_WORDS)))
        self._refresh_calibration()
        counts['personagens_ativos']=sum(g.enabled for g in self.groups.values())
        counts['altura_alvo_contorno_px']=round(self.target_height,2)
        counts['grupos']=len(self.groups)
        counts['grupos_ativos']=sum(g.enabled for g in self.groups.values())
        counts['segundos_analise']=round(time.time()-t0,2)
        self.scan_stats=dict(counts)
        return self.scan_stats

    def _refresh_calibration(self):
        """Calibrate characters from naturally upright *silhouettes*, not file/cell dimensions."""
        self.group_baselines={}
        baselines=[]
        for g in self.groups.values():
            candidates=[self.items[sid] for sid in g.ids if self.items[sid].eligible and self.items[sid].category in STANDING_MOTIONS]
            tracks=defaultdict(list)
            for it in candidates:tracks[it.category].append(it)
            # Prefer repeated walk cycles; singleton idle can have different original zoom.
            selected=tracks.get('walk',[]) if len(tracks.get('walk',[]))>=3 else tracks.get('idle',[])
            if not selected:selected=tracks.get('block',[]) or tracks.get('punch',[]) or tracks.get('hit',[])
            if not selected:
                selected=[self.items[sid] for sid in g.ids if self.items[sid].eligible and self.items[sid].height_of_subject>0]
            if not selected:continue
            heights=[float(it.height_of_subject) for it in selected if it.height_of_subject>0]
            if not heights:continue
            base=float(statistics.median(heights))
            self.group_baselines[g.key]=base
            if g.enabled:baselines.append(base)
        self.target_height=float(statistics.median(baselines)) if baselines else 0.

    def set_target_height(self,height:float):
        if not 16<=float(height)<=4096:raise ValueError('A altura precisa ficar entre 16 e 4096 pixels.')
        self.target_height=float(height)

    def _track_metrics(self,its:list[Sprite]):
        heights=[float(i.height_of_subject) for i in its if i.height_of_subject>0]
        masses=[float(i.ink_area) for i in its if i.ink_area>0]
        width=[float(i.core_width) for i in its if i.core_width>0]
        return (statistics.median(heights) if heights else 0,
                statistics.median(masses) if masses else 0,
                statistics.median(width) if width else 0)

    def _group_confidence(self,g:Group) -> str:
        items=[self.items[i] for i in g.ids]
        if len(items)<2:return 'baixa'
        tracks=len({it.track for it in items})
        scores=[it.score for it in items]
        signatures=[np.asarray(it.signature) for it in items]
        c=np.asarray(signatures); spread=float(np.median(np.linalg.norm(c-np.median(c,axis=0),axis=1)))
        if (len(items)>=3 and spread<0.42) or (statistics.median(scores)>0.8 and spread<0.65):
            return 'alta'
        if spread<0.85 or tracks==1:return 'média'
        return 'baixa'

    def _pick_reference(self,g:Group)->str:
        its=[self.items[i] for i in g.ids]
        heights=[i.height_of_subject for i in its]
        med=statistics.median(heights)
        def rank(it:Sprite):
            tokens=set(_tokenize(it.file))
            neutral=bool(tokens & {'idle','stand','standing','parado'})
            protected=bool(tokens & PROTECTED_WORDS)
            aspect=it.width_of_subject/max(1,it.height_of_subject)
            return (int(neutral)*10-int(protected)*10, -abs(it.height_of_subject-med),
                    1 if 0.22<aspect<0.95 else 0, it.score)
        return max(its,key=rank).id

    def set_reference(self,group_key:str, id_:str):
        if id_ not in self.groups[group_key].ids:
            raise ValueError('Referência não pertence a esse personagem')
        self.groups[group_key].reference=id_

    def assign_to_group(self,ids:list[str],new_group:str):
        """Manual review: move selected frames into existing or new group."""
        new_group=new_group.strip()
        if not new_group:raise ValueError('Informe o nome do grupo')
        for id_ in ids:
            it=self.items[id_]
            old=self.groups.get(it.group)
            if old and id_ in old.ids:old.ids.remove(id_)
            it.group=new_group
            # Rebind tracks to new group, preserving action semantics
            suffix=it.track.split('::',1)[-1]
            it.track=new_group+'::'+suffix
            g=self.groups.setdefault(new_group,Group(key=new_group))
            if id_ not in g.ids:g.ids.append(id_)
        for key,g in list(self.groups.items()):
            if not g.ids:del self.groups[key];continue
            if g.reference not in g.ids:g.reference=self._pick_reference(g)
            g.confidence=self._group_confidence(g)
            g.ids.sort(key=lambda i:(self.items[i].file,self.items[i].frame))

    def _track_scales(self,group:Group, strength:float=1.0) -> dict[str,float]:
        """One uniform zoom per animation; use foreground ink mass when the character is bent.

        This avoids treating a falling character's short bbox as a tiny body.
        """
        tracks=defaultdict(list)
        for sid in group.ids:
            it=self.items[sid]
            if it.eligible and it.bbox:tracks[it.track].append(it)
        if not tracks:return {}
        upright=[i for t in tracks.values() for i in t if i.category=='walk']
        if len(upright)<3:
            upright=[i for t in tracks.values() for i in t if i.category in STANDING_MOTIONS]
        if not upright:upright=[i for t in tracks.values() for i in t]
        source_height=float(self.group_baselines.get(group.key,0)) or self._track_metrics(upright)[0]
        source_ink=self._track_metrics(upright)[1]
        target=self.target_height or source_height
        desired_zoom=target/max(1.,source_height)
        factors={}
        for track,its in tracks.items():
            motion=its[0].category
            th,ta,tc=self._track_metrics(its)
            if th<=0:continue
            ratio_h=source_height/th
            ratio_mass=math.sqrt(source_ink/ta) if ta>0 and source_ink>0 else ratio_h
            if motion=='dive' and th>source_height*1.6:
                # Oversized one-pose flying kick: use silhouette height, not the massive source canvas.
                pose_ratio=ratio_h
            elif motion in STANDING_MOTIONS:
                # Neutral poses: true foreground height dominant, mass helps different art treatments.
                pose_ratio=ratio_h**0.78*ratio_mass**0.22
            else:
                # Kneeling, jumping, lying down: silhouette mass is more stable than height.
                pose_ratio=ratio_mass
            pose_ratio=max(0.20,min(2.35,pose_ratio))
            factor=(desired_zoom*pose_ratio)
            if not 0.12<=factor<=4:
                self.warnings.append(f'Escala extrema não aplicada: {group.key} / {motion} ({factor:.2f}x)')
                factor=1.
            factor=1+(factor-1)*min(1.,max(0.,strength))
            factors[track]=factor
        return factors

    def render(self,src:Image.Image,it:Sprite,factor:float, align:bool=True,
               preserve_size:bool=True, pixel_art:bool=True) -> tuple[Image.Image,dict]:
        """Preserve transparent frame canvas and natural pose; transform only opaque silhouette."""
        src=src.convert('RGBA')
        result=src.copy()
        audit={'scale_requested':round(factor,5),'scale_applied':1.,'adjusted':False,'limited':False,'warnings':[]}
        if not it.bbox or factor<=0 or abs(factor-1)<0.025:
            return result,audit
        x0,y0,x1,y1=it.bbox
        w=x1-x0;h=y1-y0
        requested_w=max(1,round(w*factor));requested_h=max(1,round(h*factor))
        # If the game's frame format is dynamic, allow expanding its transparent canvas.
        # Otherwise never change dimensions or crop body parts.
        maxfac=min(src.width/max(1,w),src.height/max(1,h)) if preserve_size else factor
        safe_factor=min(factor,maxfac)
        if safe_factor<factor-0.015:
            audit['limited']=True
            audit['warnings'].append('Escala limitada para impedir corte e preservar o tamanho do frame')
        target_w=max(1,round(w*safe_factor));target_h=max(1,round(h*safe_factor))
        if abs(target_w-w)<=1 and abs(target_h-h)<=1:
            return result,audit
        crop=src.crop((x0,y0,x1,y1))
        resized=crop.resize((target_w,target_h),RESAMPLE_NEAREST if pixel_art else RESAMPLE_SMOOTH)
        # Horizontal center and ground baseline from original pose. All frames stay in their cells.
        if align:
            cx=src.width/2
            cy=y1
            nx=round(cx-target_w/2)
            ny=round(cy-target_h)
        else:
            nx=round((x0+x1-target_w)/2)
            ny=round(y1-target_h)
        nx=max(0,min(src.width-target_w,nx))
        ny=max(0,min(src.height-target_h,ny))
        if not preserve_size:
            margin_x=max(3,round(src.width*0.025))
            margin_y=max(3,round(src.height*0.025))
            final_w=max(src.width,target_w+2*margin_x)
            final_h=max(src.height,target_h+2*margin_y)
            nx=round((final_w-target_w)/2)
            ny=max(0,final_h-max(margin_y,src.height-y1)-target_h)
        else:
            final_w,final_h=src.size
        canvas=Image.new('RGBA',(final_w,final_h),(0,0,0,0))
        canvas.alpha_composite(resized,(nx,ny))
        result.close();result=canvas
        audit['scale_applied']=round(safe_factor,5)
        audit['adjusted']=True
        if (nx==0 or ny==0 or nx+target_w==final_w or ny+target_h==final_h) and safe_factor>1:
            audit['warnings'].append('O personagem ficou próximo da borda do frame')
        return result,audit

    def _compute_factors(self,strength:float=1.) -> dict[str,float]:
        factors={}
        for g in self.groups.values():
            if not g.enabled or not g.reference:continue
            scales=self._track_scales(g,strength)
            tracks=defaultdict(list)
            for sid in g.ids:
                if self.items[sid].eligible:
                    tracks[self.items[sid].track].append(self.items[sid])
            for key,frames in tracks.items():
                base=scales.get(key,1.)
                # Isolated size errors corrected only for stable upright animations.
                median_height=statistics.median([i.height_of_subject for i in frames]) if frames else 0
                stable=(len(frames)>=4 and median_height>0 and
                        sum(abs(i.height_of_subject/median_height-1)<.11 for i in frames)>=round(.65*len(frames)))
                for it in frames:
                    factor=base
                    if stable and it.category in ('idle','walk','run') and it.height_of_subject:
                        deviation=it.height_of_subject/median_height
                        if deviation<.78 or deviation>1.28:
                            correction=max(.5,min(2.,median_height/it.height_of_subject))
                            factor*=1+(correction-1)*min(1.,max(0.,strength))
                    factors[it.id]=factor
        return factors

    def _load_frame(self,z:zipfile.ZipFile,it:Sprite)->Image.Image:
        with z.open(it.file) as stream:
            with Image.open(stream) as image:
                if it.frame:image.seek(it.frame)
                rgba=image.convert('RGBA')
                if it.cell:return rgba.crop(it.cell)
                return rgba

    def preview(self,it_id:str,strength:float=1.,pixel_art:bool=True,expand:bool|None=None):
        it=self.items[it_id]
        factors=self._compute_factors(strength)
        with zipfile.ZipFile(self.input_path,'r') as z:
            src=self._load_frame(z,it)
        result,audit=self.render(src,it,factors.get(it.id,1.),pixel_art=pixel_art,
                                 preserve_size=not (self.canvas_expand if expand is None else expand))
        return src,result,audit

    def _edit_image(self,image_bytes:bytes, its:list[Sprite],factors:dict,pixel_art:bool):
        bio=BytesIO(image_bytes)
        with Image.open(bio) as im:
            fmt=im.format or its[0].format
            if fmt=='GIF':
                frames=[];durations=[]
                for fi in range(getattr(im,'n_frames',1)):
                    im.seek(fi)
                    frame=im.convert('RGBA')
                    for it in its:
                        if it.frame==fi:
                            new,audit=self.render(frame,it,factors.get(it.id,1.),pixel_art=pixel_art)
                            frame.close();frame=new
                            it.changed=audit['adjusted'];it.audit=audit['warnings'];it.scale=audit['scale_applied']
                    frames.append(frame)
                    durations.append(im.info.get('duration',100))
                out=BytesIO()
                frames[0].save(out,format='GIF',save_all=True,append_images=frames[1:],duration=durations,loop=im.info.get('loop',0),
                               disposal=2,optimize=False,transparency=0)
                for frame in frames:frame.close()
                return out.getvalue()
            im.load();frame=im.convert('RGBA')
            cells=sorted([it for it in its if it.cell],key=lambda it:it.cell[0])
            is_fixed_strip=bool(cells and self.canvas_expand and all(it.atlas_fixed for it in cells) and
                                len(cells)==self.atlas_counts.get(its[0].file,0))
            if is_fixed_strip:
                # Rebuild entire sprite strip with a uniform per-frame canvas, keeping frame order.
                newcells=[]
                for it in cells:
                    src=frame.crop(it.cell)
                    new,audit=self.render(src,it,factors.get(it.id,1.),pixel_art=pixel_art,preserve_size=False)
                    src.close()
                    newcells.append((it,new,audit))
                cell_w=max(c.width for _,c,_ in newcells)
                cell_h=max(c.height for _,c,_ in newcells)
                rebuilt=Image.new('RGBA',(cell_w*len(newcells),cell_h))
                for idx,(it,new,audit) in enumerate(newcells):
                    bx=(idx*cell_w)+(cell_w-new.width)//2
                    by=cell_h-new.height
                    rebuilt.alpha_composite(new,(bx,by))
                    new.close()
                    it.changed=audit['adjusted'];it.audit=audit['warnings'];it.scale=audit['scale_applied']
                frame.close();frame=rebuilt
            else:
                for it in its:
                    if it.cell:
                        sub=frame.crop(it.cell)
                        edited,audit=self.render(sub,it,factors.get(it.id,1.),pixel_art=pixel_art)
                        # No expanded crop in an atlas whose layout is not proven.
                        frame.paste(edited,it.cell[:2]);sub.close();edited.close()
                    else:
                        edited,audit=self.render(frame,it,factors.get(it.id,1.),pixel_art=pixel_art,
                                                 preserve_size=not self.canvas_expand)
                        frame.close();frame=edited
                    it.changed=audit['adjusted'];it.audit=audit['warnings'];it.scale=audit['scale_applied']
            out=BytesIO()
            save_format='PNG' if fmt=='PNG' else fmt
            save_kwargs={}
            if save_format=='PNG':
                save_kwargs={'compress_level':6}
                if 'icc_profile' in im.info:save_kwargs['icc_profile']=im.info['icc_profile']
            if save_format=='WEBP':save_kwargs={'lossless':True,'quality':100}
            frame.save(out,format=save_format,**save_kwargs)
            frame.close()
            return out.getvalue()

    def _adapt_metro_fighters_script(self,data:bytes)->bytes:
        """Opt-in adapter for this exact dynamic Canvas game; normalizes display factors.

        Its loader otherwise re-scales sprite art independently and would cancel
        the physical proportions that were just established from silhouettes.
        """
        if self.game_adapter!='metro_fighters':return data
        src=data.decode('utf-8')
        anchor='\n}\n\n// faixa de profundidade:'
        if anchor not in src:
            raise RuntimeError('Adaptador: fim de loadAll não encontrado (JS desconhecido)')
        if 'SPRITEMORPH_ADAPTER_V02' in src:
            raise RuntimeError('Este jogo já possui o adaptador v0.2 aplicado. Use o ZIP original.')
        target=round(float(self.target_height),6)
        inject=(
            f'\n  // SPRITEMORPH_ADAPTER_V02: escala artística unificada via contorno alfa\n'
            f'  // Não altera lógica de combate, faixa de circulação ou movimentação.\n'
            f'  for(const ch of ["player","punk","blue","black"]){{\n'
            f'    REFS[ch]={target};\n'
            f'    for(const [key,frames] of Object.entries(IMG)){{\n'
            f'      if(!key.startsWith(ch+"_") || !Array.isArray(frames))continue;\n'
            f'      for(const f of frames){{ f.normK=1; f.snorm=1; f.lnorm=1; }}\n'
            f'    }}\n'
            f'  }}\n'
            f'}}\n\n// faixa de profundidade:'
        )
        src=src.replace(anchor,inject,1)
        # Avoid pose-specific nonuniform stretch that previously changed character width.
        for name in ['GUARD_XSQ','PUNCH_XSQ','KICK_XSQ','UPPER_USQ','DIZZY_SQ']:
            src,n=re.subn(r'(const\s+'+name+r'\s*=\s*)[-+]?\d+(?:\.\d+)?\s*;',r'\g<1>1;',src,count=1)
            if n!=1:raise RuntimeError('Adaptador: constante '+name+' não encontrada')
        return src.encode('utf-8')

    def _audit_rendered_contours(self,archive_path:Path,byfile:dict) -> dict:
        """Verify the original visible silhouette was not cropped or nonuniformly stretched."""
        checked=0;reasons=[]
        with zipfile.ZipFile(archive_path,'r') as z:
            for name,sprites in byfile.items():
                if name not in z.namelist():
                    reasons.append(f'Arquivo de sprite ausente: {name}')
                    continue
                with Image.open(BytesIO(z.read(name))) as im:
                    fmt=im.format
                    frames=getattr(im,'n_frames',1)
                    atlas=bool(self.canvas_expand and name in self.atlas_counts and
                               any(it.atlas_fixed for it in sprites))
                    cols=self.atlas_counts.get(name,0)
                    if atlas and (im.width%cols):
                        reasons.append(f'Atlas de quadros irregulares: {name}')
                        continue
                    for it in sprites:
                        if not it.bbox:continue
                        if it.frame>=frames:
                            reasons.append(f'Frame ausente: {it.id}')
                            continue
                        im.seek(it.frame)
                        frame=im.convert('RGBA')
                        if atlas:
                            ci=int(it.id.rsplit('#c',1)[-1]);x0=round(ci*im.width/cols);x1=round((ci+1)*im.width/cols)
                            content=frame.crop((x0,0,x1,im.height));frame.close();frame=content
                        elif it.cell:
                            content=frame.crop(it.cell);frame.close();frame=content
                        bbox,_,_=_alpha_stats(frame)
                        frame.close()
                        checked+=1
                        if not bbox:
                            reasons.append(f'Silhueta sumiu: {it.id}')
                            continue
                        actual_w=bbox[2]-bbox[0];actual_h=bbox[3]-bbox[1]
                        scale=it.scale if it.changed else 1.
                        expw=it.width_of_subject*scale
                        exph=it.height_of_subject*scale
                        if abs(actual_w-expw)>max(3.,.035*expw) or abs(actual_h-exph)>max(3.,.035*exph):
                            reasons.append(f'Contorno alterado ou cortado: {it.id}, esperava {expw:.0f}x{exph:.0f}, veio {actual_w}x{actual_h}')
        if reasons:
            raise RuntimeError('AUDITORIA DE SILHUETAS: '+ '; '.join(reasons[:6]))
        return {'quadros_reverificados':checked,'contornos_cortados':0,'distorcoes_de_proporcao':0}

    def export(self,output_path:str, strength:float=1.,pixel_art:bool=True,
               notify:Optional[Callable[[str,int,int],None]]=None,full_integrity:bool=True,expand:bool|None=None):
        """Streaming ZIP copy. Rewrites only edited assets; preserves all other bytes by entry payload."""
        self.cancel.clear()
        if not self.input_path:raise RuntimeError('Analise um ZIP primeiro')
        input_abs=Path(self.input_path).resolve();out_abs=Path(output_path).resolve()
        if out_abs==input_abs:raise ValueError('A exportação não pode substituir o ZIP original')
        out_abs.parent.mkdir(parents=True,exist_ok=True)
        part=out_abs.with_name(out_abs.name+'.incompleto')
        if expand is not None:self.canvas_expand=bool(expand)
        factors=self._compute_factors(strength)
        byfile=defaultdict(list)
        changing_files={self.items[sid].file for sid,factor in factors.items() if abs(factor-1)>=0.025}
        # Every cell must be present in a declared atlas before any reconstruction.
        for it in self.items.values():
            if it.file in changing_files and (it.eligible or it.atlas_fixed):
                byfile[it.file].append(it)
        for it in self.items.values():
            it.changed=False;it.scale=1.;it.audit=[]
        adapt_script=bool(changing_files and self.canvas_expand and self.adapt_engine and self.game_adapter=='metro_fighters')
        visual_integrity={}
        edited_paths=[];failed=[];info_map={}
        start=time.time()
        try:
            with zipfile.ZipFile(input_abs,'r',allowZip64=True) as zin, zipfile.ZipFile(part,'w',allowZip64=True) as zout:
                infos=zin.infolist();n=len(infos)
                for ix,info in enumerate(infos,1):
                    self._check_cancel()
                    out_info=copy.copy(info)
                    out_info.flag_bits &= ~0x08
                    try:
                        if info.is_dir():
                            zout.writestr(out_info,b'')
                        elif adapt_script and info.filename=='src/ff_game.js':
                            original=zin.read(info)
                            adjusted=self._adapt_metro_fighters_script(original)
                            zout.writestr(out_info,adjusted)
                            edited_paths.append(info.filename)
                        elif info.filename in byfile:
                            # Work with the compressed entry only, not an extracted game folder.
                            with zin.open(info,'r') as f:
                                old=f.read()
                            try:
                                changed=self._edit_image(old,byfile[info.filename],factors,pixel_art)
                                # Verify decoding, size and frame count BEFORE accepting change.
                                with Image.open(BytesIO(old)) as a,Image.open(BytesIO(changed)) as b:
                                    if (not self.canvas_expand and a.size!=b.size) or getattr(a,'n_frames',1)!=getattr(b,'n_frames',1):
                                        raise RuntimeError('Dimensão inesperada do arquivo ou quantidade de frames foi alterada')
                                    b.verify()
                                zout.writestr(out_info,changed)
                                edited_paths.append(info.filename)
                            except Exception as ex:
                                failed.append({'arquivo':info.filename,'erro':str(ex)})
                                zout.writestr(out_info,old)
                        else:
                            with zin.open(info,'r') as f, zout.open(out_info,'w',force_zip64=True) as out:
                                shutil.copyfileobj(f,out,length=1024*1024)
                    except Exception as ex:
                        raise RuntimeError(f'Falha ao copiar {info.filename}: {ex}') from ex
                    if notify and (ix%5==0 or ix==n):notify('Exportando novo ZIP',ix,n)
            # Deep audit: silhouettes must retain width/height ratio and remain uncut.
            self._check_cancel()
            if notify:notify('Auditoria de contornos dos personagens',0,1)
            visual_integrity=self._audit_rendered_contours(part,byfile)
            if notify:notify('Auditoria de contornos dos personagens',1,1)
            # Mandatory ZIP audit, exact file list/CRC validation optional for very large projects.
            self._check_cancel()
            with zipfile.ZipFile(input_abs,'r') as z1,zipfile.ZipFile(part,'r') as z2:
                a=z1.infolist();b=z2.infolist()
                if len(a)!=len(b) or [i.filename for i in a]!=[i.filename for i in b]:
                    raise RuntimeError('AUDITORIA: arquivos ou ordem da estrutura do ZIP não preservados')
                edits=set(edited_paths)
                for i,j in zip(a,b):
                    if i.filename not in edits and (i.CRC!=j.CRC or i.file_size!=j.file_size):
                        raise RuntimeError(f'AUDITORIA: arquivo original não modificado divergiu: {i.filename}')
                if full_integrity:
                    self._check_cancel()
                    if notify:notify('Auditoria de integridade ZIP',0,1)
                    corrupt=z2.testzip()
                    if corrupt:raise RuntimeError(f'AUDITORIA: CRC inválido no arquivo {corrupt}')
                    if notify:notify('Auditoria de integridade ZIP',1,1)
            part.replace(out_abs)
        except Exception:
            if part.exists():part.unlink()
            raise
        issues=[]
        for it in self.items.values():
            if it.audit:
                issues.extend({'arquivo':it.file,'sprite':it.id,'aviso':v} for v in it.audit)
        report={
            'ferramenta':'SpriteMorph Studio 0.2',
            'metodo':'contorno alfa, area da silhueta, massa visual, movimentos e atlas declarados',
            'alvo_altura_contorno_px':self.target_height,'expansao_de_quadros':self.canvas_expand,
            'adaptador_de_motor':self.game_adapter if adapt_script else 'não aplicado',
            'arquivo_original':str(input_abs),'arquivo_exportado':str(out_abs),
            'resumo_analise':self.scan_stats,
            'arquivos_editados':len(edited_paths),'frames_ajustados':sum(it.changed for it in self.items.values()),
            'imagens_preservadas_sem_alteracao':len([it for it in self.items.values() if not it.changed]),
            'falhas_de_processamento':failed,'alertas_de_auditoria':issues,
            'avisos_analise':self.warnings,'auditoria_crc_completa':bool(full_integrity),
            'auditoria_de_silhuetas':visual_integrity,
            'segundos_processamento':round(time.time()-start,2),
            'grupos':[{ 'nome':g.key,'ativo':g.enabled,'confianca':g.confidence,
                        'quantidade_frames':len(g.ids),'referencia':g.reference} for g in self.groups.values()],
            'detalhes_frames':[
                {'arquivo':it.file,'frame':it.frame,'celula':it.cell,'personagem':it.group,'movimento':it.category,
                 'contorno_original':{'largura':it.width_of_subject,'altura':it.height_of_subject,
                 'area_tinta':it.ink_area,'largura_central':round(it.core_width,1)},
                 'alterado':it.changed,'fator_escala':it.scale,'avisos':it.audit}
                for it in self.items.values() if it.eligible
            ]
        }
        report_path=out_abs.with_name(out_abs.stem+'_auditoria.json')
        with open(report_path,'w',encoding='utf-8') as file:
            json.dump(report,file,indent=2,ensure_ascii=False)
        self.last_export=str(out_abs);self.audit_report=report
        return report
