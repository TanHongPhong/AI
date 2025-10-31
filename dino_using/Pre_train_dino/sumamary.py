import os
import cv2
import numpy as np
import argparse

def imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # force 3-channel BGR
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {path}")
    return img

def maybe_resize(img, target_h=None):
    """Resize theo chiều cao với nội suy phù hợp để tránh mờ.
       - target_h None hoặc <=0: không resize (sắc nét nhất)
       - Downscale: INTER_AREA (đỡ mờ)
       - Upscale:   INTER_CUBIC (mượt)"""
    if img is None or not target_h or target_h <= 0:
        return img
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=interp)

def make_image_grid(images, cols=3, pad=10, bg=(30, 30, 30)):
    """Ghép dạng lưới:
       - Không làm biến dạng ảnh (không ép cùng kích thước) -> chỉ pad viền để hconcat/vconcat hợp lệ.
       - Các hàng được pad về cùng width trước khi vconcat."""
    if not images:
        return None

    rows = []
    max_row_w = 0

    for i in range(0, len(images), cols):
        row_imgs = images[i:i+cols]
        # Chiều cao tối đa của hàng này
        max_h = max(img.shape[0] for img in row_imgs)

        # Pad từng ảnh lên cùng chiều cao (KHÔNG resize, nên không mờ)
        padded = []
        for img in row_imgs:
            h, w = img.shape[:2]
            top = (max_h - h) // 2
            bottom = max_h - h - top
            img_pad = cv2.copyMakeBorder(img, top, bottom, 0, 0,
                                         cv2.BORDER_CONSTANT, value=bg)
            padded.append(img_pad)
            if pad > 0:
                # cột ngăn cách giữa các ảnh trong cùng hàng
                sep_col = np.full((max_h, pad, 3), bg, dtype=np.uint8)
                padded.append(sep_col)
        if pad > 0 and len(padded) > 0:
            padded = padded[:-1]  # bỏ cột pad cuối

        row = cv2.hconcat(padded)
        rows.append(row)
        max_row_w = max(max_row_w, row.shape[1])

    # Pad mỗi row về cùng width rồi thêm khoảng cách giữa các hàng
    stacked = []
    for idx, row in enumerate(rows):
        h, w = row.shape[:2]
        if w < max_row_w:
            row = cv2.copyMakeBorder(row, 0, 0, 0, max_row_w - w,
                                     cv2.BORDER_CONSTANT, value=bg)
        stacked.append(row)
        if pad > 0 and idx < len(rows) - 1:
            sep_row = np.full((pad, max_row_w, 3), bg, dtype=np.uint8)
            stacked.append(sep_row)

    grid = cv2.vconcat(stacked)
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="D:\A UEH_UNIVERSITY\RESEACH\AI - Copy\outputs\sum\cos_sim", help="Thư mục chứa ảnh output")
    ap.add_argument("--out", default="summary.png", help="Tên file tổng hợp (khuyến nghị .png để không mờ)")
    ap.add_argument("--cols", type=int, default=3, help="Số cột mỗi hàng")
    ap.add_argument("--pad", type=int, default=10, help="Khoảng cách giữa ảnh")
    ap.add_argument("--thumb_h", type=int, default=-1,
                    help="Chiều cao thumbnail. -1 hoặc 0 = không resize (sắc nét nhất)")
    ap.add_argument("--max_width", type=int, default=-1,
                    help="Giới hạn chiều rộng ảnh tổng hợp; -1 = không giới hạn")
    args = ap.parse_args()

    folder = args.folder
    out_path = os.path.join(folder, args.out)

    # Lấy danh sách ảnh (bỏ file summary cũ)
    paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))
             and "summary" not in f.lower()]
    paths.sort()

    images = []
    for p in paths:
        img = imread_color(p)
        if img is None:
            continue
        # Resize tùy chọn (nếu cần thu nhỏ để file tổng hợp không quá to)
        img = maybe_resize(img, args.thumb_h)
        images.append(img)

    if not images:
        print("[ERROR] Không có ảnh hợp lệ trong thư mục.")
        return

    grid = make_image_grid(images, cols=args.cols, pad=args.pad, bg=(30, 30, 30))
    if grid is None:
        print("[ERROR] Không thể tạo ảnh tổng hợp.")
        return

    # Tùy chọn: thu nhỏ ảnh tổng hợp nếu quá rộng (dùng INTER_AREA để không mờ)
    if args.max_width and args.max_width > 0 and grid.shape[1] > args.max_width:
        scale = args.max_width / float(grid.shape[1])
        new_h = int(round(grid.shape[0] * scale))
        grid = cv2.resize(grid, (args.max_width, new_h), interpolation=cv2.INTER_AREA)

    # Lưu PNG để tránh mờ do nén
    out_ext = os.path.splitext(out_path)[1].lower()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_ext in [".jpg", ".jpeg"]:
        # Nếu cố tình dùng JPEG, set chất lượng tối đa
        cv2.imwrite(out_path, grid,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                     int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
    else:
        # PNG: lossless
        cv2.imwrite(out_path, grid)

    print(f"[✅ DONE] Đã lưu: {out_path}")

if __name__ == "__main__":
    main()
