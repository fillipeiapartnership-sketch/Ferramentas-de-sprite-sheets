"""Chroma-screen cleanup for already segmented sprite frames (offline).

The original rembg alpha is retained as a starting point. Regions connected
with the image border are removed, while narrow contaminated areas on the
silhouette are de-spilled from nearby interior colours. Distinct internally
supported green features are protected whenever possible.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def sample_background_color(rgb: np.ndarray) -> tuple[str | None, np.ndarray, float]:
    h, w = rgb.shape[:2]
    strip = max(1, min(16, int(min(h, w)*.035)))
    border = np.concatenate((rgb[:strip].reshape(-1,3),rgb[-strip:].reshape(-1,3),
                             rgb[:, :strip].reshape(-1,3),rgb[:, -strip:].reshape(-1,3)))
    hsv = cv2.cvtColor(border.reshape(1,-1,3),cv2.COLOR_RGB2HSV)[0]
    r,g,b = (border[:,i].astype(np.int16) for i in range(3))
    bright = border.max(axis=1).astype(np.int16)
    low = border.min(axis=1).astype(np.int16)
    is_green = (g-r>18)&(g-b>16)&(g>52)&(hsv[:,1]>55)&(hsv[:,0]>=30)&(hsv[:,0]<=90)
    is_dark = (bright<58)&((bright-low)<36)
    green_conf = float(np.mean(is_green)) if border.size else 0.0
    dark_conf = float(np.mean(is_dark)) if border.size else 0.0
    if green_conf >= max(0.08, dark_conf*1.08):
        return 'green', np.median(border[is_green],axis=0).astype(np.uint8), green_conf
    if dark_conf >= 0.10:
        chosen = border[is_dark] if is_dark.any() else border
        return 'dark', np.median(chosen,axis=0).astype(np.uint8), dark_conf
    return None, np.array([0,0,0],np.uint8), 0.0


def _green_candidates(rgb: np.ndarray,key: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    kh = int(cv2.cvtColor(np.uint8([[key]]),cv2.COLOR_RGB2HSV)[0,0,0])
    delta = np.abs(hsv[...,0].astype(np.int16)-kh)
    delta = np.minimum(delta,180-delta)
    rr,gg,bb = [rgb[...,i].astype(np.float32) for i in range(3)]
    excess = gg-np.maximum(rr,bb)
    likely = (delta<=25)&(gg>47)&(excess>10)&(hsv[...,1]>38)
    stronger = (delta<=18)&(gg>65)&(excess>23)&(hsv[...,1]>60)
    return likely,stronger


def _dark_candidates(rgb: np.ndarray,key: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    key = key.astype(np.float32)
    delta = np.sqrt(np.sum((rgb.astype(np.float32) - key[None,None,:])**2, axis=2))
    bright = rgb.max(axis=2).astype(np.int16)
    low = rgb.min(axis=2).astype(np.int16)
    sat = hsv[...,1].astype(np.int16)
    # Treat near-black / low-chroma backgrounds as removable, but remain
    # conservative around legitimate dark outlines.
    likely = ((bright < 72) & (delta < 88) & ((bright-low) < 54)) | ((bright < 58) & (sat < 110))
    stronger = ((bright < 42) & (delta < 62) & ((bright-low) < 36)) | ((bright < 34) & (sat < 85))
    return likely, stronger


def _border_connected(mask: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
    if n < 2:
        return np.zeros_like(mask,bool)
    ids=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    return np.isin(labels,ids[ids>0]) if (ids>0).any() else np.zeros_like(mask,bool)


def clean_green_spill(rgba: Image.Image, original: Image.Image | None = None,
                      mode: str='auto', strength: int=85, radius: int=10) -> Image.Image:
    """Remove outside green pixels, then chroma de-spill contour contaminants.

    auto operates only if an objectively green backdrop fills >= 18% of border;
    green allows forced de-spill when the mask/background is imperfect.
    Transparent source RGB is zeroed (no green fringe on rescaling).
    """
    if mode not in ('auto','green','dark','off'):
        raise ValueError('Modo de limpeza inválido')
    strength=int(np.clip(strength,0,100)); radius=int(np.clip(radius,1,32))
    frame=np.asarray(rgba.convert('RGBA'),dtype=np.uint8).copy()
    if mode=='off' or strength==0:
        return Image.fromarray(frame,'RGBA')
    original=(original if original is not None else rgba).convert('RGB')
    if original.size!=rgba.size:
        raise ValueError('A imagem original e o sprite devem ter dimensões iguais.')
    raw=np.asarray(original,dtype=np.uint8)
    detected_kind, key, confidence = sample_background_color(raw)
    bg_kind = mode if mode in ('green','dark') else detected_kind
    if bg_kind is None:
        return Image.fromarray(frame,'RGBA')
    if mode=='auto' and confidence<(.18 if bg_kind=='green' else .14):
        return Image.fromarray(frame,'RGBA')
    alpha=frame[...,3].copy()
    likely, strong = (_green_candidates(raw,key) if bg_kind=='green' else _dark_candidates(raw,key))
    if not likely.any():
        return Image.fromarray(frame,'RGBA')
    amount=strength/100.
    # Completely remove *external* keyed pixels, even where the AI labelled
    # background as fully opaque. A small dilation bridges JPEG halo gaps.
    external=_border_connected(cv2.morphologyEx(strong.astype(np.uint8),
                                       cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0)
    # Avoid deep intrusion into genuine costume/outline pixels. Dark outlines in
    # pixel art may touch the dark backdrop visually, so flood removal for dark
    # backgrounds only targets weak/semitransparent edge pixels.
    if bg_kind=='green':
        alpha[external & strong]=0
    else:
        alpha[(external & strong) & (alpha<245)] = 0

    visible=alpha>8
    if not visible.any():
        frame[...,:3]=0;frame[...,3]=0
        return Image.fromarray(frame,'RGBA')

    dist=cv2.distanceTransform(visible.astype(np.uint8),cv2.DIST_L2,3)
    near=(dist<=radius)&visible
    labels_count,labels,stats,_ = cv2.connectedComponentsWithStats(
        (likely & visible).astype(np.uint8),connectivity=8)
    protected=np.zeros_like(visible,bool)
    internal_dirty=np.zeros_like(visible,bool)
    key_distance=np.sqrt(np.sum((raw.astype(np.float32)-key.astype(np.float32))**2,axis=2))
    for lid in range(1,labels_count):
        component=labels==lid
        area=int(stats[lid,cv2.CC_STAT_AREA])
        deep=np.count_nonzero(component&(alpha>=245)&(dist>=max(2.5,radius*.35)))
        if bg_kind=='green':
            # Legitimate green costumes/hair accessories usually have their own
            # opaque core away from the cut edge.
            exact=component&strong&(key_distance<24)
            is_stain=(3<=area<=160 and np.count_nonzero(exact)>=area*.78
                      and np.count_nonzero(component&(dist>radius*.7))>=max(2,area*.35))
            if is_stain:
                internal_dirty|=component
            elif area>=14 and deep>=min(7,max(3,area//5)):
                protected|=component
        else:
            # Black outlines are common in pixel art; only protect larger dark
            # structures with an opaque interior, and treat tiny isolated dark
            # islands as leftover background stains.
            close=component&strong&(key_distance<52)
            is_stain=(2<=area<=90 and np.count_nonzero(close)>=max(2,int(area*.72))
                      and np.count_nonzero(component&(dist>radius*.68))>=max(1,area//4))
            if is_stain:
                internal_dirty|=component
            elif area>=18 and deep>=min(8,max(3,area//4)):
                protected|=component

    candidate=(near&likely&~protected)|internal_dirty
    if not candidate.any():
        frame[...,3]=alpha
        frame[alpha==0,:3]=0
        return Image.fromarray(frame,'RGBA')

    seed=visible&~likely&(alpha>=220)&(dist>=1.4)
    if not seed.any():
        frame[...,3]=alpha
        frame[alpha==0,:3]=0
        return Image.fromarray(frame,'RGBA')
    nearest_input=(~seed).astype(np.uint8)
    _,lbl=cv2.distanceTransformWithLabels(nearest_input,cv2.DIST_L2,5,
                                          labelType=cv2.DIST_LABEL_PIXEL)
    ys,xs=np.where(seed)
    mapping=np.zeros((int(lbl.max())+1,3),dtype=np.float32)
    mapping[lbl[ys,xs]]=frame[ys,xs,:3]
    neighbor=mapping[lbl]
    observed=raw.astype(np.float32)
    if bg_kind=='green':
        nr,ng,nb=[neighbor[...,i] for i in range(3)]
        rr,gg,bb=[observed[...,i] for i in range(3)]
        excess=np.maximum(0,gg-np.maximum(rr,bb))
        seed_excess=np.maximum(0,ng-np.maximum(nr,nb))
        contamination=np.clip((excess-seed_excess-5.)/max(30.,float(key[1])-float(max(key[0],key[2]))),0,1)
        contam=contamination*amount
        active=candidate&(contam>.045)
        if active.any():
            blend=np.clip(contam*.95,.0,.94)
            blend[internal_dirty]=np.maximum(blend[internal_dirty],.98*amount)
            recovered=observed*(1-blend[...,None])+neighbor*blend[...,None]
            recovered[...,1]=np.minimum(recovered[...,1],
                       np.maximum(recovered[...,0],recovered[...,2])+
                       np.maximum(0,seed_excess)+np.maximum(2.,(excess-seed_excess)*(1-blend)))
            frame[active,:3]=np.clip(recovered[active],0,255).astype(np.uint8)
            keydiff=np.sqrt(np.sum((observed-key.astype(np.float32))**2,axis=2))
            opacity=np.clip((78.-keydiff)/78.,0,1)*contam
            erosion=active & (~internal_dirty) & ((dist<=max(2.5,radius*.4))|(alpha<235))
            alpha[erosion]=np.clip(alpha[erosion].astype(np.float32)*(1-opacity[erosion]*.98),0,255).astype(np.uint8)
    else:
        observed_luma=np.mean(observed,axis=2)
        neighbor_luma=np.mean(neighbor,axis=2)
        keydiff=np.sqrt(np.sum((observed-key.astype(np.float32))**2,axis=2))
        closeness=np.clip((72.-keydiff)/72.,0,1)
        dark_shift=np.clip((neighbor_luma-observed_luma-4.)/55.,0,1)
        contam=closeness*dark_shift*amount
        active=candidate&(contam>.05)
        if active.any():
            blend=np.clip(contam*.96,.0,.96)
            blend[internal_dirty]=np.maximum(blend[internal_dirty],.98*amount)
            recovered=observed*(1-blend[...,None])+neighbor*blend[...,None]
            frame[active,:3]=np.clip(recovered[active],0,255).astype(np.uint8)
            opacity=np.clip((52.-keydiff)/52.,0,1)*contam
            erosion=active & (~internal_dirty) & ((dist<=max(2.2,radius*.35))|(alpha<235))
            alpha[erosion]=np.clip(alpha[erosion].astype(np.float32)*(1-opacity[erosion]*.99),0,255).astype(np.uint8)
    frame[...,3]=alpha
    frame[alpha==0,:3]=0
    return Image.fromarray(frame,'RGBA')
