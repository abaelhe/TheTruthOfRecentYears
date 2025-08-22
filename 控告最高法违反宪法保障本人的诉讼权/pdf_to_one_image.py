from io import StringIO, BytesIO
from pdf2image import convert_from_path as cfp
import os, sys, cv2 as cv, numpy as np

pdffile = name_prefix = sys.argv[1]


def mark_line(img, row_pos, col_pos, color, thickness):
    print("  mark ", row_pos, col_pos)
    t, (r0, r1), (c0, c1) = thickness, row_pos, col_pos
    img[max(r0, 0) : (r0 + t), max(c0, 0) : c1, :] = color
    img[max(r1 - t, 0) : r1, max(c0, 0) : c1, :] = color
    img[max(r0, 0) : r1, max(c0, 0) : (c0 + t), :] = color
    img[max(r0, 0) : r1, max(c1 - t, 0) : c1, :] = color
    return img


def detect_row_bound(img, noise_threshold=10, min_pixels=3):
    rows, cols, chs = img.shape
    head_index, tail_index = 0, (rows - 1)
    total_bound = cols * chs * 255
    total_threshold = min_pixels * chs * noise_threshold

    row_scores = [(total_bound - img[i, :, :].sum()) for i in range(rows)]

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

    return [head_index, tail_index]


# image array processing [rows, columns, channels]
def crop_image(img, row_pos, col_pos, chs_pos, noise_threshold=10):
    cropped = img[
        row_pos[0] : row_pos[1], col_pos[0] : col_pos[1], chs_pos[0] : chs_pos[1]
    ]
    return cropped


# create colored image array from BytesIO object
def image_from_byte_io(bytes_io):
    bytes_io.seek(0)
    file_bytes = np.asarray(bytearray(bytes_io.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    return img


def image_from_pdffile(pdf_file):
    # name_fmt = '%s.%d%s'
    images = cfp(pdf_file)
    images_num = len(images)
    bsios = [BytesIO() for i in range(images_num)]
    rets = [images[i].save(bsios[i], "JPEG") for i in range(images_num)]
    imgs = [image_from_byte_io(bsios[i]) for i in range(images_num)]
    return imgs, images_num


def pdf_to_image0(pdf_file, noise_threshold=10, min_pixels=3, line_height=0):
    imgs, images_num = image_from_pdffile(pdffile)
    if images_num < 1:
        return []

    color, thickness = (255, 0, 0), 5
    rows, cols, chs = imgs[0].shape
    chs_pos, e_args, m_args = [0, chs], ((0, 0, 255), 2), (color, thickness)
    print("Cropping: rows:%s, cols:%s, " % (rows, cols))

    crop_col_bound = (0, cols)

    crop_row_head = detect_row_bound(imgs[0], noise_threshold, min_pixels)
    crop_row_tail = detect_row_bound(imgs[-1], noise_threshold, min_pixels)
    crop_row_tail = [
        max(0, crop_row_tail[0] - line_height),
        min(rows, crop_row_tail[1] + crop_row_head[0]),
    ]
    crop_row_head = [0, crop_row_head[1]]

    crop_args = [(crop_row_head, crop_col_bound)]

    for i in imgs[1:-1]:
        crop_row_bound = detect_row_bound(i, noise_threshold, min_pixels)
        crop_row_bound = [max(0, crop_row_bound[0] - line_height), crop_row_bound[1]]
        crop_args.append((crop_row_bound, crop_col_bound))

    crop_args.append((crop_row_tail, crop_col_bound))

    m_imgs = []
    i_imgs = []

    for i in range(len(imgs)):
        img = imgs[i]
        i_rows, i_cols, i_chs = img.shape

        crop_arg = crop_args[i]
        m_img = mark_line(img.copy(), (0, i_rows), (0, i_cols), *e_args)
        m_img = mark_line(m_img, *crop_arg, *m_args)
        m_imgs.append(m_img)

        i_img = crop_image(img.copy(), *crop_arg, [0, i_chs])
        i_imgs.append(i_img)

    m_concated = cv.vconcat(m_imgs)
    i_concated = cv.vconcat(i_imgs)
    return m_concated, i_concated


def pdf_to_image1(pdf_file, row_ratio=0.0729, col_ratio=0.0):
    imgs, images_num = image_from_pdffile(pdffile)
    if images_num < 1:
        return []

    color, thickness = (255, 0, 0), 5
    rows, cols, chs = imgs[0].shape
    row_delta, col_delta = int(rows * row_ratio), int(cols * col_ratio)
    chs_pos, e_args, m_args = [0, chs], ((0, 0, 255), 2), (color, thickness)
    print("Cropping: rows:%s, cols:%s, " % (rows, cols))

    crop_arg_head = ((0, rows - row_delta), (col_delta, cols - col_delta))
    crop_arg_other = ((row_delta, rows - row_delta), (col_delta, cols - col_delta))
    crop_arg_tail = ((row_delta, rows), (col_delta, cols - col_delta))

    # import pdb;pdb.set_trace()
    m_concated, i_concated = None, None
    if images_num == 1:
        m_head = mark_line(imgs[0].copy(), (0, rows), (0, cols), *e_args)
        m_head = mark_line(m_head, *crop_arg_other, *m_args)
        m_concated = m_head
        i_head = crop_image(imgs[0].copy(), (0, rows), (0, cols), chs_pos)
        i_concated = i_head
    elif images_num > 1:
        m_head = mark_line(imgs[0].copy(), (0, rows), (0, cols), *e_args)
        m_head = mark_line(m_head, *crop_arg_head, *m_args)
        m_other = [
            mark_line(i.copy(), (0, rows), (0, cols), *e_args) for i in imgs[1:-1]
        ]
        m_other = [mark_line(i, *crop_arg_other, *m_args) for i in m_other]
        m_tail = mark_line(imgs[-1].copy(), (0, rows), (0, cols), *e_args)
        m_tail = mark_line(m_tail, *crop_arg_tail, *m_args)
        m_imgs = [m_head] + m_other + [m_tail]
        m_concated = cv.vconcat(m_imgs)

        i_head = crop_image(imgs[0].copy(), *crop_arg_head, chs_pos)
        i_other = [crop_image(i.copy(), *crop_arg_other, chs_pos) for i in imgs[1:-1]]
        i_tail = crop_image(imgs[-1].copy(), *crop_arg_tail, chs_pos)
        i_imgs = [i_head] + i_other + [i_tail]
        i_concated = cv.vconcat(i_imgs)
    return m_concated, i_concated


# vertical concat all images in imgs_processed
m_concated, i_concated = pdf_to_image0(
    pdffile, noise_threshold=10, min_pixels=3, line_height=0
)
cv.imwrite(pdffile + "_0marked.jpg", m_concated)
cv.imwrite(pdffile + "_0.jpg", i_concated)

m_concated, i_concated = pdf_to_image1(pdffile, row_ratio=0.0729, col_ratio=0.0)

# save the processed image array
cv.imwrite(pdffile + "_1marked.jpg", m_concated)
cv.imwrite(pdffile + "_1.jpg", i_concated)
