"""Local sprite extraction engine. No internet requests in this module.
Video decode uses every frame. Background segmentation is provided by rembg on local ONNX.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter
from green_cleanup import clean_green_spill
from uniform_black import cut_uniform_dark


@dataclass
class Frame:
    index: int
    timestamp: float
    original: str
    sprite: str = ""
    selected: bool = False
    reason: str = ""
    roi: tuple[int,int,int,int] | None = None


@dataclass
class Group:
    name: str
    source: str
    fps: float
    folder: str
    frames: list[Frame] = field(default_factory=list)
    width: int = 0
    height: int = 0


def imwrite_unicode(path: str | Path, img: np.ndarray, ext=".jpg", quality=96):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() == ".jpg" else []
    ok, data = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"Erro ao codificar {path}")
    Path(path).write_bytes(data.tobytes())


def significant_signature(frame_bgr: np.ndarray):
    # Keep subtle silhouette differences: comparison uses the entire image, but
    # measures changed-pixel area rather than only global hash similarity.
    im = cv2.resize(frame_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(im, (3, 3), .4)


def _clamp_roi(roi, width: int, height: int):
    if roi is None:
        return None
    x, y, w, h = [int(round(v)) for v in roi]
    x = max(0, min(x, width - 1)); y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x)); h = max(1, min(h, height - y))
    return (x, y, w, h)


def _signature_with_roi(frame_bgr: np.ndarray, roi=None):
    roi = _clamp_roi(roi, frame_bgr.shape[1], frame_bgr.shape[0])
    if roi is None:
        return significant_signature(frame_bgr)
    x, y, w, h = roi
    subject = frame_bgr[y:y+h, x:x+w]
    if subject.size == 0:
        return significant_signature(frame_bgr)
    canvas = np.zeros_like(frame_bgr)
    canvas[y:y+h, x:x+w] = subject
    return significant_signature(canvas)


def _create_tracker():
    constructors = []
    if hasattr(cv2, 'legacy'):
        constructors.extend([
            getattr(cv2.legacy, 'TrackerCSRT_create', None),
            getattr(cv2.legacy, 'TrackerKCF_create', None),
            getattr(cv2.legacy, 'TrackerMIL_create', None),
        ])
    constructors.extend([
        getattr(cv2, 'TrackerCSRT_create', None),
        getattr(cv2, 'TrackerKCF_create', None),
        getattr(cv2, 'TrackerMIL_create', None),
    ])
    for maker in constructors:
        if maker is not None:
            try:
                return maker()
            except Exception:
                pass
    return None


def is_similar(sig: np.ndarray, prior: np.ndarray, affected_limit: float) -> bool:
    delta = cv2.absdiff(sig, prior)
    pixel_delta = np.max(delta, axis=2)
    # Bitrate artefacts around static video pixels shouldn't count as a new pose.
    affected = np.count_nonzero(pixel_delta > 13) / pixel_delta.size
    large = np.count_nonzero(pixel_delta > 35) / pixel_delta.size
    return affected <= affected_limit and large <= affected_limit * .45 + .0004


def candidate_indexes(signatures: list[np.ndarray], sensitivity: int = 2):
    """Conservative duplicate rejection. Return indices in chronological order."""
    limits = {0: .0001, 1: .0015, 2: .0035, 3: .0065, 4: .012}
    limit = limits[max(0, min(4, sensitivity))]
    kept_sig = []
    indexes = []
    for i, sig in enumerate(signatures):
        if not any(is_similar(sig, prev, limit) for prev in kept_sig):
            indexes.append(i)
            kept_sig.append(sig)
    return indexes


def detect_repeat_period(signatures: list[np.ndarray], min_period: int = 4) -> int | None:
    """Find convincing repeat cycles even when compression/lighting drift.

    A real repeated motion forms a distinct trough in the lag-similarity curve,
    often another trough at twice the lag. Requiring exact colour equality
    incorrectly rejected real footage with noisy frames. We still reject
    nearly static footage, random motion and ambiguous isolated minima.
    """
    n=len(signatures)
    if n<min_period*2+2:
        return None
    small=np.stack([cv2.resize(s,(32,32),interpolation=cv2.INTER_AREA)
                    for s in signatures]).astype(np.float32)
    adjacent=np.mean(np.abs(small[1:]-small[:-1]),axis=(1,2,3))
    movement=float(np.median(adjacent))
    if movement<.20:
        return None
    max_lag=min(n//2,180)
    scores={}
    for lag in range(min_period,max_lag+1):
        count=min(n-lag,max(12,lag*2),110)
        picks=np.linspace(0,n-lag-1,count,dtype=int)
        err=np.mean(np.abs(small[picks]-small[picks+lag]),axis=(1,2,3))
        scores[lag]=float(np.median(err))
    if not scores:
        return None
    best=min(scores.values())
    # Conservative exact-match path remains for clean synthetic sprite loops.
    strict=[(lag,score) for lag,score in scores.items()
            if score<=max(.27*movement,min(.65*movement,best*1.8))
            and score<=movement*.65]
    if strict:
        return min(strict,key=lambda v:v[0])[0]

    # Real videos contain codec noise, small colour drift and frame interpolation.
    # A periodic valley must stand out from nearby WRONG-phase comparisons,
    # and its second harmonic must recur when there are enough frames.
    viable=[]
    for lag,score in scores.items():
        if score>movement*1.65 or score>best*1.20:
            continue
        neighbors=[scores[k] for k in range(max(min_period,lag-6),min(max_lag,lag+6)+1)
                   if abs(k-lag)>=3]
        if len(neighbors)<3 or score>float(np.median(neighbors))*.77:
            continue
        harmonic=scores.get(lag*2)
        if harmonic is not None and harmonic>score*1.32:
            continue
        viable.append((lag,score))
    if not viable:
        return None
    # Favor the earliest genuine repeat, never the 2x/3x harmonic.
    return min(viable,key=lambda v:v[0])[0]


def best_cycle_window(signatures: list[np.ndarray], period: int) -> tuple[int,int]:
    """Choose a complete contiguous cycle whose last pose joins first smoothly.

    Avoid clips starting mid-motion. Use source frames on a fixed canvas,
    not the moving tracker ROI. Return (start, period).
    """
    n=len(signatures)
    period=int(period)
    if period<2 or period>n:
        raise ValueError('Período do ciclo inválido')
    # A single complete cycle can be returned without a second matching one.
    if n<2*period:
        return (0,period)
    small=np.stack([cv2.resize(s,(32,32),interpolation=cv2.INTER_AREA)
                    for s in signatures]).astype(np.float32)
    def diff(i,j):return float(np.mean(np.abs(small[i]-small[j])))
    quality=[]
    for start in range(n-2*period+1):
        seam=diff(start,start+period-1)
        phase=np.median([diff(j,j+period) for j in range(start,start+period)])
        # Prioritize a seamless loop; phase agreement avoids mismatched cycles.
        quality.append((seam+0.20*float(phase),start))
    return (min(quality)[1],period)


def suggest_cycle_window_for_group(group: Group) -> tuple[int,int] | None:
    signatures=[]
    for fr in group.frames:
        raw=cv2.imread(fr.original)
        if raw is None:
            raise ValueError(f'Quadro original ausente: {fr.original}')
        signatures.append(significant_signature(raw))
    period=detect_repeat_period(signatures)
    return best_cycle_window(signatures,period) if period else None


def suggest_cycle_for_group(group: Group) -> int | None:
    signatures=[]
    for fr in group.frames:
        raw=cv2.imread(fr.original)
        if raw is None:
            raise ValueError(f'Quadro original ausente: {fr.original}')
        signatures.append(significant_signature(raw))
    return detect_repeat_period(signatures)


def foreground_signature(image: Image.Image) -> np.ndarray:
    im = image.convert('RGBA')
    a = np.asarray(im.getchannel('A'))
    yy, xx = np.where(a > 20)
    if len(xx):
        im = im.crop((max(0, int(xx.min())-2), max(0, int(yy.min())-2),
                      min(im.width, int(xx.max())+3), min(im.height, int(yy.max())+3)))
    # normalized silhouette fingerprint, with RGB so blinks and facial changes survive
    im.thumbnail((92, 92), Image.Resampling.LANCZOS)
    frame = Image.new('RGBA', (96, 96), (0, 0, 0, 0))
    frame.paste(im, ((96-im.width)//2, (96-im.height)//2))
    return np.asarray(frame).copy()


def foreground_similar(a: np.ndarray, b: np.ndarray, sensitivity: int):
    # Preserve details: a contour difference of even a few pixels matters.
    alpha_a, alpha_b = a[:, :, 3], b[:, :, 3]
    alpha_delta = cv2.absdiff(alpha_a, alpha_b)
    changed_alpha = np.count_nonzero(alpha_delta > 40) / alpha_delta.size
    both = (alpha_a > 50) & (alpha_b > 50)
    rgb_delta = np.max(cv2.absdiff(a[:, :, :3], b[:, :, :3]), axis=2)
    changed_color = np.count_nonzero((rgb_delta > 34) & both) / rgb_delta.size
    limit = {0: .0002, 1: .0015, 2: .003, 3: .006, 4: .012}[max(0,min(4,sensitivity))]
    return changed_alpha < limit and changed_color < limit * 1.3


def _keep_guided_subject(alpha_image: Image.Image, roi) -> Image.Image:
    """Use ROI to choose a subject, NOT to crop or resize the inference frame.

    Most clips have only one subject. In that case leave the model mask
    completely unchanged. If multiple disconnected large subjects are present,
    choose the component overlapping the user's guide and keep nearby details.
    The guide cannot alter where the sprite is placed on the source canvas.
    """
    alpha = np.asarray(alpha_image, dtype=np.uint8)
    h, w = alpha.shape
    roi = _clamp_roi(roi, w, h)
    if roi is None:
        return alpha_image
    x, y, rw, rh = roi
    # Only significant components participate in subject identification.
    binary = (alpha > 70).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1,n)]
    if not areas:
        return alpha_image
    min_area = max(20, int(max(areas) * 0.075))
    big = [i for i in range(1,n) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]
    if len(big) <= 1:
        return alpha_image
    # Distinguish objects, never crop the actual pixels to the tracker box.
    pad = max(10, round(max(rw, rh) * .18))
    l = max(0,x-pad); t = max(0,y-pad)
    r = min(w,x+rw+pad); b = min(h,y+rh+pad)
    hits=[]
    for i in big:
        count = int(np.count_nonzero(labels[t:b,l:r] == i))
        hits.append((count,i))
    count, target = max(hits)
    if count < max(12, int(stats[target, cv2.CC_STAT_AREA] * .03)):
        # Uncertain tracking. Preserve the segmentation rather than amputating.
        return alpha_image
    tx,ty,tw,th,ta = [int(v) for v in stats[target]]
    # Keep smaller disconnected character features near the principal component.
    margin = max(12, round(max(tw,th)*.20))
    near = (max(0,tx-margin), max(0,ty-margin), min(w,tx+tw+margin), min(h,ty+th+margin))
    keep_ids={target}
    for i in range(1,n):
        if i==target:continue
        cx,cy,cw,ch,ca=[int(v) for v in stats[i]]
        if cx>=near[0] and cy>=near[1] and cx+cw<=near[2] and cy+ch<=near[3] and ca<=ta*.6:
            keep_ids.add(i)
    # Preserve soft anti-aliased pixels around connected components.
    safe = np.isin(labels,list(keep_ids)).astype(np.uint8)
    safe = cv2.dilate(safe, np.ones((7,7),np.uint8),iterations=1)
    new_alpha = np.where(safe>0,alpha,0).astype(np.uint8)
    return Image.fromarray(new_alpha, 'L')


def remove_bg_file(raw_path: str, save_path: str, session, edge: int=0,
                   cleanup_mode: str='auto', cleanup_strength: int=85,
                   cleanup_radius: int=10, roi=None):
    """Stable full-frame foreground inference; optional ROI only selects identity.

    v6-v9 applied rembg to a moving ROI crop. A wobbling tracker shifted
    context and mask boundaries each frame, causing visible artificial jitter.
    Always infer from identically sized source frames with a fixed canvas.
    """
    with Image.open(raw_path) as f:
        original = f.convert('RGB')
    # Special case for clearly uniform black backgrounds. Do not use a
    # changing ROI crop or trust a weak AI matte for black pixel-art outlines.
    keyed=(cut_uniform_dark(original, roi) if cleanup_mode in ('auto','dark') else None)
    if keyed is not None:
        base_img=keyed
    else:
        from rembg import remove, new_session
        # Lazy model load: plain black footage needs no ONNX weights.
        if isinstance(session,dict):
            if session['instance'] is None:
                session['instance']=new_session(session['model'])
            active_session=session['instance']
        else:
            active_session=session
        reduced = original.copy()
        reduced.thumbnail((1280,1280), Image.Resampling.LANCZOS)
        result=remove(reduced,session=active_session,alpha_matting=False).convert('RGBA')
        if result.size != original.size:
            result=result.resize(original.size,Image.Resampling.LANCZOS)
        alpha=_keep_guided_subject(result.getchannel('A'),roi)
        if edge:
            k=min(15,abs(edge)*2+1)
            alpha=alpha.filter(ImageFilter.MinFilter(k) if edge<0 else ImageFilter.MaxFilter(k))
        base=np.asarray(original.convert('RGBA')).copy()
        model_rgb=np.asarray(result)
        a=np.asarray(alpha)
        halo=(a>0)&(a<245)
        base[halo,:3]=model_rgb[halo,:3]
        base[:,:,3]=a
        base_img=Image.fromarray(base,'RGBA')
    dest=Path(save_path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    cache=dest.parent.parent/'recortes_base_ia'/dest.name
    cache.parent.mkdir(parents=True,exist_ok=True)
    base_img.save(cache)
    cleaned=(base_img if keyed is not None else
             clean_green_spill(base_img,original,cleanup_mode,cleanup_strength,cleanup_radius))
    cleaned.save(dest)
    return foreground_signature(cleaned)


def reapply_cleanup(group: Group, indexes: list[int], mode: str='auto',
                    strength: int=85, radius: int=10,
                    callback=None, cancelled=None):
    """Reprocess GREEN spill without rerunning rembg or touching source JPGs.

    Older V1 projects do not have cached bases. A safety copy of the last PNG
    is created the first time their cleanup settings are changed.
    """
    report = callback or (lambda p,s:None)
    stopping = cancelled or (lambda:False)
    changed=[]
    for n,idx in enumerate(indexes):
        if stopping(): raise InterruptedError('Limpeza cancelada.')
        fr=group.frames[idx]
        sprite=Path(fr.sprite)
        if not sprite.is_file():continue
        cache=sprite.parent.parent/'recortes_base_ia'/sprite.name
        cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.is_file():
            from shutil import copy2
            copy2(sprite,cache)
        with Image.open(cache) as img:
            base=img.convert('RGBA')
        with Image.open(fr.original) as original:
            cleaned=clean_green_spill(base,original,mode,strength,radius)
        # Temp+replace so interrupted saves never corrupt sprite PNGs.
        target=sprite.with_name(sprite.stem+'.tmp.png')
        cleaned.save(target)
        target.replace(sprite)
        changed.append(idx)
        report(round((n+1)*100/max(1,len(indexes))),
               f'Limpando halos verdes: {n+1}/{len(indexes)}')
    return changed


def process_video(video: str, base_dir: str, model: str, sensitivity: int=2,
                  edge: int=0, callback: Callable[[int,str],None] | None = None,
                  cancelled: Callable[[],bool] | None = None,
                  cleanup_mode: str='auto', cleanup_strength: int=85,
                  cleanup_radius: int=10, initial_roi=None) -> Group:
    """Decode AND save every source frame; segment candidate key poses only."""
    report = callback or (lambda p, s: None)
    stopping = cancelled or (lambda: False)
    src = Path(video)
    dest = Path(base_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rawdir, spritedir = dest/'frames_originais', dest/'sprites_rgba'
    rawdir.mkdir(exist_ok=True)
    spritedir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src.name}. Tente converter para MP4 H.264.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (0.1 < fps < 240):
        fps = 24.0
    estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, signatures = [], []
    count = 0
    tracker = None
    last_roi = None
    try:
        while True:
            if stopping():
                raise InterruptedError('Processamento cancelado.')
            ok, frame = cap.read()
            if not ok:
                break
            if initial_roi is not None:
                if tracker is None:
                    tracker = _create_tracker()
                    last_roi = _clamp_roi(initial_roi, frame.shape[1], frame.shape[0])
                    if tracker is not None and last_roi is not None:
                        try:
                            tracker.init(frame, tuple(last_roi))
                        except Exception:
                            tracker = None
                elif tracker is not None:
                    try:
                        ok_track, box = tracker.update(frame)
                    except Exception:
                        ok_track, box = False, None
                    if ok_track and box is not None:
                        last_roi = _clamp_roi(box, frame.shape[1], frame.shape[0])
            path = rawdir/f'frame_{count:06}.png'
            imwrite_unicode(path, frame, ext='.png')
            frames.append(Frame(index=count, timestamp=count/fps, original=str(path), roi=last_roi))
            # Identical source coordinates for cycle recognition too; moving ROI
            # borders must never appear as fake changes in the animation.
            signatures.append(significant_signature(frame))
            count += 1
            if count % 12 == 0:
                report(int(32 * count / max(count,estimate)), f'Extraindo todos os quadros: {count} / {estimate if estimate>0 else "?"}')
    finally:
        cap.release()
    if count == 0:
        raise ValueError('O vídeo não contém quadros legíveis.')
    report(35, f'{count} quadros extraídos. ' + ('Rastreando personagem e ' if initial_roi is not None else '') + 'identificando poses distintas...')
    cycle = detect_repeat_period(signatures)
    cycle_start = 0
    if cycle:
        cycle_start, cycle = best_cycle_window(signatures,cycle)
        indexes = list(range(cycle_start,cycle_start+cycle))
        report(37, f'Ciclo de {cycle} quadros localizado em {cycle_start+1}–{cycle_start+cycle}; preservando sequência contínua.')
    else:
        # NEVER globally de-duplicate non-periodic motion. Different temporal
        # gaps played at a uniform FPS cause stutter even with perfect masks.
        indexes = list(range(count))
        report(37, 'Nenhum ciclo confiável: preservando todos os quadros consecutivos, sem saltos temporais.')
    sig_set = set(indexes)
    for i, fr in enumerate(frames):
        if i not in sig_set:
            fr.reason = 'fora do ciclo contínuo selecionado'
    report(38, f'{len(indexes)} poses candidatas. Preparando remoção de fundo...')
    # Load the AI model only if a frame needs it; a uniform black screen
    # can be cut offline without loading ONNX.
    session={'instance':None,'model':model}
    kept = []
    for n, idx in enumerate(indexes):
        if stopping():
            raise InterruptedError('Processamento cancelado.')
        fr = frames[idx]
        out = spritedir/f'sprite_{idx:06}.png'
        cut_sig = remove_bg_file(fr.original, str(out), session, edge,
                                 cleanup_mode, cleanup_strength, cleanup_radius, fr.roi)
        fr.sprite = str(out)
        # Keep every consecutive frame of the chosen segment. Deleting
        # visually similar frames here would unevenly speed up the animation.
        fr.selected = True
        kept.append(cut_sig)
        if n % 2 == 0 or n == len(indexes)-1:
            report(40+int(60*(n+1)/max(1,len(indexes))),
                   f'Removendo fundo: {n+1}/{len(indexes)} · {len(kept)} sprites principais')
    grp = Group(name=src.stem, source=str(src), fps=fps, folder=str(dest), frames=frames,
                width=width, height=height)
    (dest/'projeto.json').write_text(json.dumps({
        'source': str(src), 'fps': fps, 'frames': count, 'model':model,
        'selected_indexes': [f.index for f in frames if f.selected],
        'created':datetime.now().isoformat(), 'cycle_length': cycle,
        'cycle_start': cycle_start, 'preserve_temporal_spacing': True,
        'guided_tracking': bool(initial_roi is not None),
        'stable_full_frame_segmentation': True,
        'rois': {str(f.index): list(f.roi) for f in frames if f.roi is not None},
        'cleanup':{'mode':cleanup_mode,'strength':cleanup_strength,'radius':cleanup_radius}
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    report(100, f'Concluído! {len(kept)} sprites em {len(frames)} quadros.')
    return grp


def process_additional(group: Group, indexes: list[int], model: str, edge:int=0,
                       callback=None, cancelled=None, cleanup_mode='auto',
                       cleanup_strength=85, cleanup_radius=10):
    report = callback or (lambda p,s:None)
    stopping = cancelled or (lambda:False)
    session={'instance':None,'model':model}
    outputs = []
    for i, idx in enumerate(indexes):
        if stopping():
            raise InterruptedError('Processamento cancelado.')
        fr = group.frames[idx]
        dst = str(Path(group.folder)/'sprites_rgba'/f'sprite_{fr.index:06}.png')
        remove_bg_file(fr.original, dst, session, edge,
                       cleanup_mode, cleanup_strength, cleanup_radius, fr.roi)
        outputs.append((idx, dst))
        report(round(100*(i+1)/max(1,len(indexes))), f'Processando quadro extra {i+1}/{len(indexes)}')
    return outputs


def alpha_bbox(img: Image.Image, cutoff: int=18):
    alpha=np.asarray(img.getchannel('A'))
    yy,xx=np.where(alpha>cutoff)
    return (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)) if xx.size else None


def _torso_anchor(img: Image.Image):
    """Estimate stable upper-body anchor, ignoring walking feet/arms extremes.

    This is used by the optional stabilization mode. Every rendered frame
    still shares one uniform scale and one fixed output cell size.
    """
    a=np.asarray(img.getchannel('A'),dtype=np.uint8)
    yy,xx=np.where(a>40)
    if xx.size==0:
        return img.width/2.,img.height/2.
    left,right=float(xx.min()),float(xx.max()+1)
    top,bottom=float(yy.min()),float(yy.max()+1)
    height=max(1.,bottom-top)
    # Scan the visually stable upper body, omitting moving legs and hands.
    sample=(a>110)&(np.arange(img.height)[:,None]>=top+height*.22)&(np.arange(img.height)[:,None]<=top+height*.58)
    xx2=np.where(sample)[1]
    if xx2.size < 8:
        return (left+right)/2.,top+height*.48
    # Median of scanline widths helps avoid a single swinging arm shifting x.
    centers=[]
    for y in range(max(0,round(top+height*.25)),min(img.height,round(top+height*.56))):
        coords=np.flatnonzero(sample[y])
        if coords.size>=4:
            centers.append((float(np.quantile(coords,.25))+float(np.quantile(coords,.75)))/2)
    return (float(np.median(centers)) if centers else float(np.median(xx2))),top+height*.48


def _safe_bbox(image: Image.Image):
    """Slight overrun keeps semitransparent hair and shoe pixels intact."""
    found = alpha_bbox(image, cutoff=5)
    if found is None:
        return (0, 0, image.width, image.height)
    return (max(0, found[0]-2), max(0, found[1]-2),
            min(image.width, found[2]+2), min(image.height, found[3]+2))


def geometry(paths: list[str], cellw=384, cellh=512, center=False,
             auto_fit=True):
    """Calculate a SINGLE alignment/scale for the animation's grid.

    For frames with the same source resolution the entire animation uses
    one unchanging transform (including walk bob, arm swing, and leg swing).
    For heterogeneous source resolutions, a stable factor is calculated per
    source-size group from the MEDIAN character height of that group.
    Every pose in the same group shares its factor, preventing per-frame zoom
    jitter. The same aspect-ratio-preserving transform serves preview/export.

    center=True can stabilize torso motion; it never arbitrarily changes the
    width independently from height.
    """
    if not paths:
        raise ValueError('Não há sprites selecionados.')
    cellw,cellh=int(cellw),int(cellh)
    if min(cellw,cellh)<=0:
        raise ValueError('As dimensões de cada grid devem ser maiores que zero.')
    boxes=[];anchors={};sizes=set();entries=[]
    for path in paths:
        path=str(path)
        with Image.open(path) as source:
            rgba=source.convert('RGBA')
            size=rgba.size
            bbox=_safe_bbox(rgba)
            torso=_torso_anchor(rgba)
        sizes.add(size)
        boxes.append(bbox)
        anchors[path]=torso
        entries.append({'path':path,'size':size,'bbox':bbox,'torso':torso})

    pad=max(2,round(min(cellw,cellh)*.035))
    if len(sizes)==1:
        # Essential: don't resize a standing pose, jumping pose, or a bent
        # knee separately based on its changing silhouette. That destroys
        # natural motion and introduces new jitter.
        outer=(min(b[0] for b in boxes),min(b[1] for b in boxes),
               max(b[2] for b in boxes),max(b[3] for b in boxes))
        maxw,maxh=outer[2]-outer[0],outer[3]-outer[1]
        if center:
            xs=[(b[0]-anchors[e['path']][0],b[2]-anchors[e['path']][0]) for e,b in zip(entries,boxes)]
            ys=[(b[1]-anchors[e['path']][1],b[3]-anchors[e['path']][1]) for e,b in zip(entries,boxes)]
            ranges=(min(v[0] for v in xs),max(v[1] for v in xs),
                    min(v[0] for v in ys),max(v[1] for v in ys))
            maxw,maxh=ranges[1]-ranges[0],ranges[3]-ranges[2]
        else:
            ranges=None
        scale=min((cellw-2*pad)/max(1,maxw),(cellh-2*pad)/max(1,maxh))
        return {'mode':'torso' if center else 'fixed',
                'center':bool(center),'outer':outer,'ranges':ranges,
                'anchors':anchors,'pad':pad,'scale':scale,
                'width':cellw,'height':cellh,'auto_fit':bool(auto_fit),
                'source_sizes':len(sizes),'group_scale':{str(next(iter(sizes))):1.0}}

    # Different resolution / PNG trim sizes: derive a single normalized
    # character scale for each source size, not for each individual frame.
    groups={}
    for e in entries:
        groups.setdefault(e['size'],[]).append(e)
    group_heights={size:float(np.median([max(1,e['bbox'][3]-e['bbox'][1])
                                          for e in members]))
                   for size,members in groups.items()}
    reference_height=float(np.median(list(group_heights.values())))
    # Limit extreme outliers so a wrongly segmented tiny foreground cannot
    # grow 20x and wreck a whole sheet. User can fix its mask and retry.
    factors={size:float(np.clip(reference_height/max(1.,h),0.25,4.0))
             if auto_fit else 1.0 for size,h in group_heights.items()}
    # Anchor once per source-size group: never chase swinging arms or raised
    # feet at each individual frame. In the optional torso mode each frame's
    # own detected anchor is used to compensate actual camera drift.
    group_anchors={size:(float(np.median([e['torso'][0] for e in members])),
                         float(np.median([e['torso'][1] for e in members])))
                   for size,members in groups.items()}
    transforms={}
    xmin=ymin=float('inf')
    xmax=ymax=float('-inf')
    for e in entries:
        size,bbox=e['size'],e['bbox']
        anchor=e['torso'] if center else group_anchors[size]
        factor=factors[size]
        x1=(bbox[0]-anchor[0])*factor
        x2=(bbox[2]-anchor[0])*factor
        y1=(bbox[1]-anchor[1])*factor
        y2=(bbox[3]-anchor[1])*factor
        xmin=min(xmin,x1);xmax=max(xmax,x2)
        ymin=min(ymin,y1);ymax=max(ymax,y2)
        transforms[e['path']]={'bbox':bbox,'factor':factor,'anchor':anchor,
                                'size':size,'world':(x1,y1,x2,y2)}
    extentw=max(1.,xmax-xmin);extenth=max(1.,ymax-ymin)
    scale=min((cellw-2*pad)/extentw,(cellh-2*pad)/extenth)
    return {'mode':'mixed','center':bool(center),'auto_fit':bool(auto_fit),
            'transforms':transforms,'world_bounds':(xmin,ymin,xmax,ymax),
            'scale':scale,'width':cellw,'height':cellh,'pad':pad,
            'source_sizes':len(sizes),
            'group_scale':{f'{key[0]}x{key[1]}':round(val,6)
                           for key,val in factors.items()}}


def render_cell(path: str, geom: dict) -> Image.Image:
    path=str(path)
    with Image.open(path) as source:
        rgba=source.convert('RGBA')
    cw,ch=geom['width'],geom['height']
    dst=Image.new('RGBA',(cw,ch),(0,0,0,0))
    mode=geom.get('mode','torso' if geom.get('center') else 'fixed')
    if mode=='mixed':
        t=geom['transforms'][path]
        # Crop only the character, retaining a small alpha margin, and place
        # it in the common world coordinate system.
        part=rgba.crop(t['bbox'])
        world=t['world']
        world_bounds=geom['world_bounds']
        scale=geom['scale']
        dest_left=(cw-(world_bounds[2]-world_bounds[0])*scale)/2.
        dest_top=(ch-(world_bounds[3]-world_bounds[1])*scale)/2.
        x0=round(dest_left+(world[0]-world_bounds[0])*scale)
        y0=round(dest_top+(world[1]-world_bounds[1])*scale)
        x1=round(dest_left+(world[2]-world_bounds[0])*scale)
        y1=round(dest_top+(world[3]-world_bounds[1])*scale)
        w,h=max(1,x1-x0),max(1,y1-y0)
        if part.size!=(w,h):
            part=part.resize((w,h),Image.Resampling.LANCZOS)
        # Transparent paste, NO mask: preserve actual source RGBA.
        dst.paste(part,(x0,y0))
        return dst
    if mode=='torso':
        ax,ay=geom['anchors'][path]
        x1,x2,y1,y2=geom['ranges']
        ox,oy=int(round(-x1)),int(round(-y1))
        w,h=max(1,int(np.ceil(x2-x1))),max(1,int(np.ceil(y2-y1)))
        crop=Image.new('RGBA',(w,h),(0,0,0,0))
        crop.paste(rgba,(round(ox-ax),round(oy-ay)))
        rgba=crop
    else:
        # Stable original coordinates for every frame in the video.
        rgba=rgba.crop(geom['outer'])
    w=max(1,round(rgba.width*geom['scale']))
    h=max(1,round(rgba.height*geom['scale']))
    if (w,h)!=rgba.size:
        rgba=rgba.resize((w,h),Image.Resampling.LANCZOS)
    dst.paste(rgba,((cw-w)//2,(ch-h)//2))
    return dst


def export_group(group: Group, out_root: str, indexes: list[int], cellw=384, cellh=512,
                 columns=6, spacing=0, center=False, report=None, auto_fit=True):
    paths=[group.frames[i].sprite for i in indexes]
    if not paths or any(not p or not Path(p).exists() for p in paths):
        raise ValueError('Selecione sprites processados. Use "Processar selecionados" primeiro.')
    base=Path(out_root)
    base.mkdir(parents=True,exist_ok=True)
    sequence=base/'sprites'
    sequence.mkdir(exist_ok=True)
    lay=geometry(paths,cellw,cellh,center,auto_fit)
    # Do not exceed the common 8192 px atlas limit.
    columns=max(1,min(int(columns),max(1,8192//(cellw+spacing))))
    max_rows=max(1,8192//(cellh+spacing))
    per_sheet=columns*max_rows
    manifest=[]
    for page_index,begin in enumerate(range(0,len(indexes),per_sheet),1):
        slice_indexes=indexes[begin:begin+per_sheet]
        rows=math.ceil(len(slice_indexes)/columns)
        used_cols=min(columns,len(slice_indexes))
        sheetw=used_cols*cellw+max(0,used_cols-1)*spacing
        sheeth=rows*cellh+max(0,rows-1)*spacing
        sheet=Image.new('RGBA',(sheetw,sheeth),(0,0,0,0))
        sheetname=f'sprite_sheet_{page_index:03}.png'
        for j,idx in enumerate(slice_indexes):
            ordinal=begin+j
            cell=render_cell(group.frames[idx].sprite,lay)
            filename=f'frame_{ordinal+1:04}.png'
            cell.save(sequence/filename)
            col,row=j%columns,j//columns
            x,y=col*(cellw+spacing),row*(cellh+spacing)
            sheet.paste(cell,(x,y))
            manifest.append({'name':filename,'source_frame':group.frames[idx].index,
                             'source_time_s':round(group.frames[idx].timestamp,4),
                             'sheet':sheetname,'rect':{'x':x,'y':y,'w':cellw,'h':cellh}})
            if report:
                report(round((ordinal+1)*100/len(indexes)),f'Exportando {ordinal+1}/{len(indexes)}')
        sheet.save(base/sheetname)
    (base/'atlas.json').write_text(json.dumps({
        'group':group.name,'source':group.source,'source_fps':group.fps,
        'cell_size':{'width':cellw,'height':cellh},'columns':columns,
        'spacing':spacing,'stable_body_anchor':center,'fixed_coordinates':not center,
        'auto_fit_proportional':bool(auto_fit),
        'source_resolution_groups':lay['source_sizes'],
        'scale_by_resolution':lay['group_scale'],
        'frames':manifest
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return len(manifest),page_index


def load_project(project_file: str) -> Group:
    """Restore extracted frames from a current or V1 project folder."""
    proj=Path(project_file)
    if proj.is_dir():proj=proj/'projeto.json'
    if not proj.is_file(): raise FileNotFoundError('Arquivo projeto.json não encontrado.')
    metadata=json.loads(proj.read_text(encoding='utf-8'))
    folder=proj.parent
    raws=sorted([*((folder/'frames_originais').glob('frame_*.png')),
                 *((folder/'frames_originais').glob('frame_*.jpg'))],
                key=lambda p:int(p.stem.split('_')[-1]))
    if not raws: raise ValueError('Os frames_originais não foram encontrados.')
    fps=float(metadata.get('fps') or 12)
    if fps<=0:fps=12
    selected=set(metadata.get('selected_indexes') or ())
    roi_map = metadata.get('rois') or {}
    items=[]
    for index,raw in enumerate(raws):
        try:frame_number=int(raw.stem.split('_')[-1])
        except (ValueError, IndexError):frame_number=index
        sprite=folder/'sprites_rgba'/f'sprite_{frame_number:06}.png'
        ready=sprite.is_file()
        roi = roi_map.get(str(frame_number))
        roi = tuple(roi) if isinstance(roi, (list, tuple)) and len(roi)==4 else None
        items.append(Frame(index=frame_number,timestamp=frame_number/fps,
                           original=str(raw),sprite=str(sprite) if ready else '',
                           selected=frame_number in selected and ready,
                           reason='projeto recuperado', roi=roi))
    with Image.open(raws[0]) as im:w,h=im.size
    return Group(name=folder.name,source=metadata.get('source',''),fps=fps,
                 folder=str(folder),frames=items,width=w,height=h)


def reprocess_stable_existing(group: Group, indexes: list[int], model: str, edge: int=0,
                              callback=None, cancelled=None, cleanup_mode='auto',
                              cleanup_strength=85, cleanup_radius=10):
    """Upgrade v6-v9 cropped masks to full-frame inference, with rollback safety.

    Previously retouched sprites and their IA base caches are backed up before
    replacement. If cancelled or interrupted, originals are restored.
    """
    from shutil import copy2
    report = callback or (lambda p,s:None)
    stopped = cancelled or (lambda:False)
    # Preflight every source first to avoid a partial update for missing frames.
    for idx in indexes:
        fr = group.frames[idx]
        if not Path(fr.original).is_file():
            raise FileNotFoundError(f'Frame original ausente: {fr.original}')
        if not fr.sprite or not Path(fr.sprite).is_file():
            raise FileNotFoundError(f'Sprite ausente: {fr.index+1}')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup=Path(group.folder)/f'backup_antes_correcao_tremor_{stamp}'
    originals=[]
    for idx in indexes:
        fr=group.frames[idx]
        sprite=Path(fr.sprite)
        base=sprite.parent.parent/'recortes_base_ia'/sprite.name
        sprite_backup=backup/'sprites_rgba'/sprite.name
        base_backup=backup/'recortes_base_ia'/base.name
        sprite_backup.parent.mkdir(parents=True,exist_ok=True)
        copy2(sprite,sprite_backup)
        had_base=base.is_file()
        if had_base:
            base_backup.parent.mkdir(parents=True,exist_ok=True)
            copy2(base,base_backup)
        originals.append((sprite,base,sprite_backup,base_backup,had_base))
    session={'instance':None,'model':model}
    try:
        for n,idx in enumerate(indexes):
            if stopped():
                raise InterruptedError('Reprocessamento cancelado. Sprites anteriores restaurados.')
            fr=group.frames[idx]
            remove_bg_file(fr.original,fr.sprite,session,edge,cleanup_mode,
                           cleanup_strength,cleanup_radius,fr.roi)
            report(round(100*(n+1)/max(1,len(indexes))),
                   f'Recorte estável: {n+1}/{len(indexes)}')
        if stopped():
            raise InterruptedError('Reprocessamento cancelado. Sprites anteriores restaurados.')
    except BaseException:
        for sprite,base,sprite_backup,base_backup,had_base in originals:
            copy2(sprite_backup,sprite)
            if had_base:
                copy2(base_backup,base)
            elif base.exists():
                base.unlink()
        raise
    return str(backup)
