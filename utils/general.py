import numpy as np
from PIL import Image

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]


def convert_image_to_rgb(image, format):
    if isinstance(image, torch.tensor):
        image = image.cpu().numpy()
    
    if format == 'BGR':
        image = image[:, :, [2,1,0]]
    elif format == 'YUV-BT.601':
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image *= 255.0
    else:
        if format == 'L':
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.assarray(Image.fromarray(image, mode = format).convert('RGB'))
    return image