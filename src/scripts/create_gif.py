import os
import cv2
import sys
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    asr_image_dir = sys.argv[1]
    vsr_image_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    asr_image_files = sorted(os.listdir(asr_image_dir))
    vsr_image_files = sorted(os.listdir(vsr_image_dir))

    for asr_image_file, vsr_image_file in tqdm(zip(asr_image_files, vsr_image_files)):
        asr_image_path = os.path.join(asr_image_dir, asr_image_file)
        vsr_image_path = os.path.join(vsr_image_dir, vsr_image_file)
        dest_path = os.path.join(output_dir, os.path.basename(asr_image_file))

        epoch = asr_image_path.split('epoch')[-1].split('.')[0]

        asr_img = cv2.imread(asr_image_path)
        vsr_img = cv2.imread(vsr_image_path)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (125,40)
        fontScale              = 1
        fontColor              = (0,0,0)
        thickness              = 3
        lineType               = 1

        cv2.putText(
            asr_img,
            f"ASR -- Epoch {epoch}",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType
        )

        cv2.putText(
            vsr_img,
            f"VSR -- Epoch {epoch}",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType
        )

        img = np.hstack((asr_img[:, :-140, :], vsr_img[:, :-50, :]))
        h,w,c = img.shape

        cv2.imwrite(dest_path, img)

    for i in range(10):
        new_dest_path = dest_path.replace(".png", f"_{str(i).zfill(2)}.png")
        cv2.imwrite(new_dest_path, img)

    for i in range(10):
        new_new_dest_path = new_dest_path.replace(".png", f"_{str(i).zfill(2)}.png")
        blank_img = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
        cv2.imwrite(new_new_dest_path, blank_img)

    os.system(f"convert -delay 30 -loop 0 {output_dir}/*.png ./my.gif")
    # os.system("rm -rf ./to_gif/")
