import numpy as np

def set_intrinsic(cam_info):
    width = cam_info.width
    height = cam_info.height
    K = np.array(cam_info.K).reshape(3,3)
    return K, width, height