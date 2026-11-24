"""Uniform dark/black background keyer for animation frames.

Conservative opt-in via classification. Protects dark line art by locating the
colored silhouette first, then retaining a narrow contour rim. Runs fully
locally with OpenCV; no model weights or network needed for this special case.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def is_uniform_dark_background(rgb: np.ndarray) -> bool:
    h,w=rgb.shape[:2]
    if min(h,w)<30:
        return False
    strip=max(2,min(12,round(min(h,w)*0.013)))
    b=np.concatenate((rgb[:strip].reshape(-1,3),rgb[-strip:].reshape(-1,3),
                      rgb[:,:strip].reshape(-1,3),rgb[:,-strip:].reshape(-1,3)))
    border_med=np.median(b,axis=0)
    delta=np.max(np.abs(b.astype(np.float32)-border_med[None,:]),axis=1)
    if max(border_med)>34 or np.mean(delta<=16)<.96:
        return False
    # Avoid blank frames and videos with dark, full-frame scenery.
    return True


def cut_uniform_dark(original: Image.Image, roi=None, rim_radius=None) -> Image.Image | None:
    """Return an RGBA sprite only for reliably flat near-black backgrounds.

    Preserves inner black linework and a narrow external dark outline. Rejects
    scenes touching image borders and backgrounds which aren't flat black.
    """
    rgb=np.array(original.convert('RGB'),dtype=np.uint8)
    h,w=rgb.shape[:2]
    if not is_uniform_dark_background(rgb):
        return None
    base=np.concatenate((rgb[:3].reshape(-1,3),rgb[-3:].reshape(-1,3),
                         rgb[:,:3].reshape(-1,3),rgb[:,-3:].reshape(-1,3)))
    key=np.median(base,axis=0).astype(np.float32)
    v=rgb.astype(np.float32)-key
    color_delta=np.sqrt(np.sum(v*v,axis=2))
    intensity=np.max(rgb,axis=2)
    foreground=(color_delta>38.0)&(intensity>27)
    m=foreground.astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<2:
        return None
    by_area=sorted(range(1,n),key=lambda lid:stats[lid,cv2.CC_STAT_AREA],reverse=True)
    if int(stats[by_area[0],cv2.CC_STAT_AREA])<max(200,int(w*h*.004)):
        return None
    # Independent hands/shoes can be detached by the darker outline.
    subject=np.zeros((h,w),dtype=np.uint8)
    strongest=int(stats[by_area[0],cv2.CC_STAT_AREA])
    picked=0
    for idx in by_area:
        area=int(stats[idx,cv2.CC_STAT_AREA]);x,y,cw,ch=[int(v) for v in stats[idx,:4]]
        if area<max(36,strongest*.00015):
            continue
        # Exclude spurious far-away colored items if guided ROI is available.
        if roi is not None:
            rx,ry,rw,rh=[int(v) for v in roi]
            pad=int(max(rw,rh)*.15)
            if x+cw <rx-pad or x>rx+rw+pad or y+ch<ry-pad or y>ry+rh+pad:
                continue
        subject[labels==idx]=255
        picked+=area
    if picked <max(200,w*h*.004):
        return None
    # Fail safe: sprite needs a little black margin for its outline.
    bounds=cv2.boundingRect(cv2.findNonZero(subject))
    x,y,bw,bh=bounds
    minmargin=max(4,int(round(min(w,h)*.006)))
    if min(x,y,w-x-bw,h-y-bh)<minmargin:
        return None
    if picked > w*h*.82:
        return None
    contours,_=cv2.findContours(subject,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outline=np.zeros_like(subject)
    cv2.drawContours(outline,contours,-1,255,-1)
    rad=(int(np.clip(round(min(w,h)*.008),8,10)) if rim_radius is None
         else int(np.clip(rim_radius,2,12)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1))
    outline=cv2.dilate(outline,kernel,iterations=1)
    alpha=cv2.GaussianBlur(outline,(3,3),.6)
    result=np.dstack([rgb,alpha])
    result[alpha==0,:3]=0
    return Image.fromarray(result,'RGBA')
