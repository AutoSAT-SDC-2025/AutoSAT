import numpy as np
import cv2 as cv

def apply_affine(img, rotation):
    points = np.array([[0,0,1], [img.shape[1], img.shape[0], 1], [0, img.shape[0], 1], [img.shape[1], 0, 1]])
    A = np.vstack([rotation, [0,0,1]])
    tpoints = (A@points.T).T
    x = -np.min(tpoints[:,0])
    y = -np.min(tpoints[:,1])
    T = np.array([[1,0,x],[0,1,y]],dtype=np.float64)
    T_h = np.vstack([T, [0,0,1]])
    tpoints = (T_h@A@points.T).T
    _x = int(np.max(tpoints[:, 0]))
    _y = int(np.max(tpoints[:, 1]))
    A = (T_h@A)[:2,:3]
    img = cv.warpAffine(img, A, (_x, _y))
    print(T_h)
    return img, np.array([[1,0,-x],[0,1,-y]],dtype=np.float64)


def rotate_img(img, rotation):
    rotation = np.hstack([rotation, [[0],[0]]]).astype(np.float32)
    img, T = apply_affine(img, rotation)
    return img, T 
