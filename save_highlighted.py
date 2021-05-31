import cv2
import multiprocessing as mp
import os
import numpy as np
from functools import partial

def highlight_and_save(image_name, src="./images", mask="./saliency", dest="./highlighted", lowerBound=10):
    try:
        img = cv2.imread(os.path.join(src, image_name))
        mask = cv2.imread(os.path.join(mask, image_name)) # 3 channels
        mask[:,:,1][mask[:,:,0]<=lowerBound] = 100  # To see the non salient results too
        mask[:,:,2][mask[:,:,0]<=lowerBound] = 100  # To see the non salient results too
        masked_image = (img*(mask/255)).astype(np.uint8)
        cv2.imwrite(os.path.join(dest, image_name), masked_image)
    except:
        print("FAILED FOR", image_name)

if __name__=="__main__":    
    base_path = os.path.join(os.curdir, "examples", "SALICON")  # need to change
    img_list = os.listdir(os.path.join(base_path, "images"))
    print(len(img_list))
    pool = mp.Pool(processes=12)  # Pool size will be 12 on my personal computer

    partial_highlight_and_save = partial(highlight_and_save,
                                            src=os.path.join(base_path, "images"),
                                            mask=os.path.join(base_path, "saliency"),
                                            dest=os.path.join(base_path, "highlighted")
                                        )
    pool.map(partial_highlight_and_save, img_list)