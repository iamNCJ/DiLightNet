import numpy as np
import rembg
from PIL import Image


def rm_bg(img, use_sam=False):
    img = Image.fromarray(img)
    img = img.resize((512, 512))
    output = rembg.remove(img)
    mask = np.array(output)[:, :, 3]

    # use sam for mask refinement
    if use_sam:
        session = rembg.new_session('sam', sam_model='sam_vit_h_4b8939')
        bool_mask = mask > 0
        y1, y2, x1, x2 = (
            np.nonzero(bool_mask)[0].min(),
            np.nonzero(bool_mask)[0].max(),
            np.nonzero(bool_mask)[1].min(),
            np.nonzero(bool_mask)[1].max()
        )
        output = rembg.remove(img, session=session, sam_prompt=[
            {'type': 'rectangle', 'label': 1, 'data': [x1, y1, x2, y2]}
        ])
        mask = np.array(output)[:, :, 3]

    return output, mask
