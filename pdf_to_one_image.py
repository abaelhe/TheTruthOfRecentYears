from io import StringIO, BytesIO
from pdf2image import convert_from_path as cfp
import os, sys, cv2 as cv, numpy as np



def mark_line(img, bound_args, color, thickness):
    print("  mark: %r " % (bound_args,))
    t, (r0, r1, c0, c1) = thickness, bound_args
    r0, r1, c0, c1 = max(r0, 0), r1,  max(c0, 0), c1
    img[r0:(r0+t),       c0:c1,       :] = color
    img[max(r1-t, 0):r1, c0:c1,       :] = color
    img[r0:r1,           c0:(c0 + t), :] = color
    img[r0:r1,  max(c1-t,0):c1,       :] = color
    return img


def detect_bound(img, noise_threshold=10, min_pixels=3):
    rows, cols, chs = img.shape
    row_s_max = cols * chs * 255
    col_s_max = rows * chs * 255
    total_threshold = min_pixels * chs * noise_threshold

    row_scores =[(row_s_max -img[i, :, :].sum()) for i in range(rows)]
    col_scores =[(col_s_max -img[:, i, :].sum()) for i in range(cols)]
    head_index, tail_index = 0, (rows - 1)
    hi, ti = head_index, tail_index
    while hi < rows:
        if row_scores[hi] > total_threshold:
            head_index = max(0, hi - 1)
            break
        hi += 1

    while ti > 0:
        if row_scores[ti] > total_threshold:
            tail_index = ti + 1
            break
        ti -= 1

    left_index, right_index = 0, (cols - 1)
    li, ri = left_index, right_index
    while li < cols:
        if col_scores[li] > total_threshold:
            left_index = max(0, li - 1)
            break
        li += 1

    while ri > 0:
        if col_scores[ri] > total_threshold:
            right_index = ri + 1
            break
        ri -= 1

    bound_args = [head_index, tail_index, left_index, right_index]
    print("crop bound: %s,\t%s,\t%s,\t%s" % (
        head_index, tail_index, left_index, right_index))
    return bound_args


# image array processing [rows, columns, channels]
def crop_image(img, crop_args, chs_pos):
    ca = crop_args
    cropped = img[ ca[0]:ca[1], ca[2]:ca[3], chs_pos[0]:chs_pos[1] ]
    return cropped


# create colored image array from BytesIO object
def image_from_byte_io(bytes_io):
    bytes_io.seek(0)
    file_bytes = np.asarray(bytearray(bytes_io.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    return img


def image_from_pdffile(pdf_file):
    # name_fmt = '%s.%d%s'
    if not os.path.isfile(pdf_file):
        raise SystemError("Doesn't exist or is not a file: %r" % pdf_file)
    images = cfp(pdf_file)
    images_num = len(images)
    bsios = [BytesIO() for i in range(images_num)]
    rets = [images[i].save(bsios[i], "JPEG") for i in range(images_num)]
    imgs = [image_from_byte_io(bsios[i]) for i in range(images_num)]
    return imgs, images_num


def pdf_to_image(pdf_file, noise_threshold=10, min_pixels=3, line_height=0, width_ratio=1.00):
    imgs, images_num = image_from_pdffile(pdf_file)
    if images_num < 1:
        return []

    color, thickness = (255, 0, 0), 5
    rows, cols, chs = imgs[0].shape
    chs_pos, e_args, m_args = [0, chs], ((0, 0, 255), 2), (color, thickness)
    print("Cropping: rows:%s, cols:%s, " % (rows, cols))
    
    img_bounds, left_min, right_max =[], cols, 0
    for i in range(len(imgs)):
       bound = detect_bound(imgs[i], noise_threshold, min_pixels)
       if bound[2] < left_min:
           left_min = bound[2]
       if bound[3] > right_max:
           right_max = bound[3]
       img_bounds.append(bound)

    width_margin = min(left_min, (cols - right_max))
    width_margin = int(width_margin * width_ratio)

    crop_head = img_bounds[0]
    crop_head[2] = width_margin
    crop_head[3] = cols -1 - width_margin

    crop_tail = img_bounds[-1]
    crop_tail[0] = max(crop_tail[0] - line_height,     0)
    crop_tail[1] = min(crop_tail[1] + crop_head[0], rows)
    crop_tail[2], crop_tail[3] = crop_head[2], crop_head[3]

    crop_head[0] = 0

    crop_args = [crop_head]

    for i in range(1, images_num-1):
        crop_bound = img_bounds[i]
        crop_bound[0] =max(0, crop_bound[0] - line_height)
        crop_bound[1] =crop_bound[1]
        crop_bound[2], crop_bound[3] = crop_head[2], crop_head[3]
        crop_args.append(crop_bound)

    crop_args.append(crop_tail)

    m_imgs = []
    i_imgs = []

    for i in range(len(imgs)):
        img = imgs[i]
        i_rows, i_cols, i_chs = img.shape
        crop_arg = crop_args[i]
        m_img = mark_line(img.copy(), (0, i_rows, 0, i_cols), *e_args)
        m_imgs.append(m_img)

    for i in range(len(imgs)):
        m_img = m_imgs[i]
        crop_arg = crop_args[i]
        mark_line(m_img, crop_arg, *m_args)

    for i in range(len(imgs)):
        img = imgs[i]
        crop_arg = crop_args[i]
        i_img = crop_image(img, crop_arg, [0, i_chs])
        i_imgs.append(i_img)

    m_concated = cv.vconcat(m_imgs)
    i_concated = cv.vconcat(i_imgs)
    return m_concated, i_concated


def main():
    if len(sys.argv) < 2:
        raise SystemError("Usage: python  pdf_to_image.py <PDF_FILE>  | [JPEG_QUALITY]")
    pdf_file = name_prefix = sys.argv[1]
    jpg_quality=100
    if len(sys.argv)> 2 and sys.argv[2].isdigit():
        jpg_quality = int(sys.argv[2])

    # vertical concat all images in imgs_processed
    m_concated, i_concated = pdf_to_image(
        pdf_file, noise_threshold=10, min_pixels=3, line_height=0
    )
    cv.imwrite(pdf_file + "_marked.jpg", m_concated, (cv.IMWRITE_JPEG_QUALITY, jpg_quality))
    cv.imwrite(pdf_file + ".jpg", i_concated, (cv.IMWRITE_JPEG_QUALITY, jpg_quality))

if "__main__" == __name__:
    main()
