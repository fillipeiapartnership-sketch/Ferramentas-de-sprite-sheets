"""Checks click-select cleanup without Qt or downloading AI weights."""
import numpy as np
import cv2
from smart_retouch import retouch_click


def run():
    # A thin contaminated border outside a filled character.
    rgba=np.zeros((100,100,4),np.uint8)
    rgba[25:80,30:74]=[100,40,180,255]
    rgba[50:55,27:30]=[45,145,45,200]
    prev=rgba.copy()
    result,message,n=retouch_click(rgba,28,52,19,18,'auto')
    assert n>0,(message,n)
    assert result[52,28,3]==0, 'green edge residue not erased'
    assert np.array_equal(result[50,50],rgba[50,50]), 'person recolored erroneously'
    assert np.array_equal(rgba,prev), 'in-place mutation prevents undo'
    print('Contorno contaminado: removido em 1 clique, corpo preservado.')

    # A red stain fully inside blue clothing should preserve the alpha.
    rgba=np.zeros((100,100,4),np.uint8)
    rgba[20:85,20:85]=[35,70,200,255]
    rgba[50:54,50:54]=[220,25,35,255]
    result,message,n=retouch_click(rgba,51,51,15,22,'auto')
    assert n>0 and result[51,51,3]==255,(message,n)
    assert np.mean(result[50:54,50:54,2])>np.mean(rgba[50:54,50:54,2]),'stain not recolored'
    assert np.array_equal(result[35,35],rgba[35,35]), 'unrelated shirt pixel changed'
    print('Mancha interna: corrigida sem criar buracos.')

    # Non-green background should also work with explicit erase click.
    rgba=np.zeros((90,90,4),np.uint8)
    rgba[15:70,25:70]=[200,90,45,255]
    rgba[40:44,20:25]=[200,180,110,255]
    result,message,n=retouch_click(rgba,22,42,20,25,'erase')
    assert n>0 and result[42,22,3]==0, (message,n)
    assert result[42,45,3]==255
    print('Mancha com cor diferente de verde: remoção por clique OK.')

    # Wrong click on a broad red uniform torso should be rejected.
    rgba=np.zeros((120,120,4),np.uint8)
    rgba[15:110,15:110]=[185,40,50,255]
    result,message,n=retouch_click(rgba,60,60,70,35,'erase')
    assert n==0 and np.array_equal(result,rgba),message
    print('Proteção contra apagar parte grande do corpo: OK.')

    # Transparent background already needs no removal.
    result,message,n=retouch_click(rgba,2,2,15,25,'auto')
    assert n==0 and np.array_equal(result,rgba)
    print('Clique na transparência: não altera PNG.')

if __name__=='__main__':run()
