import PIL.Image as pimg

"按照缩放的边长对图片等比例缩放，并转成正方形居中"


def scale_img(img, scale_side):
    # "获得图片宽高"
    w1, h1 = img.size
    # print(w1,h1)
    # "根据最大边长缩放,图像只会被缩小，不会变大"
    # "当被缩放的图片宽和高都小于缩放尺寸的时候，图像不变"
    img.thumbnail((scale_side, scale_side))
    # "获得缩放后的宽高"
    w2, h2 = img.size
    # print(w2,h2)
    # "获得缩放后的比例"
    # s1 = w1 / w2
    # s2 = h1 / h2
    # s = (s1 + s2) / 2
    # "新建一张scale_side*scale_side的空白黑色背景图片"
    bg_img = pimg.new("RGB", (scale_side, scale_side), (0, 0, 0))
    # "根据缩放后的宽高粘贴图像到背景图上"
    if w2 == scale_side:
        bg_img.paste(img, (0, int((scale_side - h2) / 2)))
    elif w2 == scale_side:
        bg_img.paste(img, (int((scale_side - w2) / 2), 0))
    # "原图比缩放后的图要小的时候"
    else:
        bg_img.paste(img, (int((scale_side - w2) / 2), (int((scale_side - h2) / 2))))
    return bg_img
