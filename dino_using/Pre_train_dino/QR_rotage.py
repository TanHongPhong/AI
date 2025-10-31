# align_flip_translate_qr.py
import cv2, os, numpy as np

# ======= CHỈ SỬA 3 DÒNG NÀY =======
IMG_REF = r"D:\path\to\image_A.jpg"   # Ảnh chuẩn
IMG_MOV = r"D:\path\to\image_B.jpg"   # Ảnh cần căn chỉnh
OUT_DIR = r"D:\path\to\outputs"
# ==================================

os.makedirs(OUT_DIR, exist_ok=True)

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect_qr_corners(img_bgr):
    qr = cv2.QRCodeDetector()
    quads = []
    try:
        ok, decoded, points, _ = qr.detectAndDecodeMulti(img_bgr)
        if ok and points is not None:
            for quad in points:
                if quad is None or len(quad) != 4: continue
                quads.append(order_points(quad))
    except Exception:
        pass
    if not quads:
        _, quad, _ = qr.detectAndDecode(img_bgr)
        if quad is not None and len(quad) == 4:
            quads.append(order_points(quad))
    if not quads: return None
    areas = [cv2.contourArea(q.astype(np.float32)) for q in quads]
    return quads[int(np.argmax(areas))]

def mse(a, b):
    return float(np.mean(np.sum((a - b)**2, axis=1)))

def main():
    A = cv2.imread(IMG_REF)
    B = cv2.imread(IMG_MOV)
    if A is None: raise FileNotFoundError(IMG_REF)
    if B is None: raise FileNotFoundError(IMG_MOV)

    hA, wA = A.shape[:2]

    cA = detect_qr_corners(A)
    cB = detect_qr_corners(B)
    if cA is None or cB is None:
        print("Không phát hiện đủ QR ở cả hai ảnh.")
        return

    # ===== Ứng viên 0°: x' = x + t0 =====
    t0 = np.mean(cA - cB, axis=0)           # (tx, ty)
    cB_to_A_0 = cB + t0                     # corners dự đoán trên A
    err0 = mse(cB_to_A_0, cA)
    M0 = np.array([[1, 0, t0[0]], [0, 1, t0[1]]], dtype=np.float32)

    # ===== Ứng viên 180°: x' = -x + t180 =====
    t180 = np.mean(cA + cB, axis=0)         # vì -cB + t = cA
    cB_to_A_180 = (-cB) + t180
    err180 = mse(cB_to_A_180, cA)
    M180 = np.array([[-1, 0, t180[0]], [0, -1, t180[1]]], dtype=np.float32)

    use_180 = err180 < err0
    M_best = M180 if use_180 else M0
    angle = 180 if use_180 else 0
    t_used = t180 if use_180 else t0
    err_used = min(err0, err180)

    # Warp B -> khung A trong MỘT BƯỚC (xoay 0/180 + dịch)
    B_aligned = cv2.warpAffine(B, M_best, (wA, hA), flags=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(A, 0.5, B_aligned, 0.5, 0)

    # Lưu
    os.makedirs(OUT_DIR, exist_ok=True)
    baseA = os.path.splitext(os.path.basename(IMG_REF))[0]
    baseB = os.path.splitext(os.path.basename(IMG_MOV))[0]
    cv2.imwrite(os.path.join(OUT_DIR, f"{baseB}_aligned_{angle}.jpg"), B_aligned)
    cv2.imwrite(os.path.join(OUT_DIR, f"{baseA}_{baseB}_overlay_{angle}.jpg"), overlay)

    # Log
    np.set_printoptions(precision=5, suppress=True)
    print(f"Chọn: {'180°' if use_180 else '0°'}")
    print(f"t_used = ({t_used[0]:.2f}, {t_used[1]:.2f})  |  err0={err0:.5f}, err180={err180:.5f}")
    print("Affine M (B -> A):\n", M_best)
    print(f"Đã lưu vào: {OUT_DIR}")

if __name__ == "__main__":
    main()
