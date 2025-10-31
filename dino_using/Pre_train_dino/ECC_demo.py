# ecc_align_demo_fixed.py
import cv2, numpy as np, os

# ========================= CONFIG (sửa ở đây) =========================
TEMPLATE_PATH = r"D:\A UEH_UNIVERSITY\RESEACH\AI\dino_using\img\WIN_20251021_02_22_44_Pro.jpg"   # ảnh chuẩn
MOVING_PATH   = r"D:\A UEH_UNIVERSITY\RESEACH\AI\dino_using\img\WIN_20251021_02_22_15_Pro.jpg"     # ảnh cần căn chỉnh
OUTDIR        = r"D:\A UEH_UNIVERSITY\RESEACH\AI\outputs\sum"   # thư mục xuất kết quả

MOTION = "affine"   # "translation" | "euclidean" | "affine" | "homography"
ALPHA  = 0.5        # độ mờ khi chồng ảnh (0..1): overlay = T*ALPHA + M*(1-ALPHA)
# =====================================================================

def to_gray_f32(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0

def ecc_align(template_bgr, moving_bgr, motion='affine', iters=100, eps=1e-6, roi_mask=None):
    H, W = template_bgr.shape[:2]
    if moving_bgr.shape[:2] != (H, W):
        moving_bgr = cv2.resize(moving_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    tm = to_gray_f32(template_bgr)
    mv = to_gray_f32(moving_bgr)

    motion_map = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean'  : cv2.MOTION_EUCLIDEAN,
        'affine'     : cv2.MOTION_AFFINE,
        'homography' : cv2.MOTION_HOMOGRAPHY
    }
    mt = motion_map[motion]

    Wmat = np.eye(3, dtype=np.float32) if mt==cv2.MOTION_HOMOGRAPHY else np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    try:
        cc, Wmat = cv2.findTransformECC(tm, mv, Wmat, mt, criteria, inputMask=roi_mask)
    except cv2.error as e:
        print("[WARN] ECC fail, dùng identity. Lý do:", e)
        cc = -1.0
        Wmat = np.eye(3, dtype=np.float32) if mt==cv2.MOTION_HOMOGRAPHY else np.eye(2,3, dtype=np.float32)

    if mt == cv2.MOTION_HOMOGRAPHY:
        moving_aligned = cv2.warpPerspective(
            moving_bgr, Wmat, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
        ones = np.ones((H, W), np.uint8)*255
        valid = cv2.warpPerspective(
            ones, Wmat, (W, H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        moving_aligned = cv2.warpAffine(
            moving_bgr, Wmat, (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
        ones = np.ones((H, W), np.uint8)*255
        valid = cv2.warpAffine(
            ones, Wmat, (W, H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    return moving_aligned, valid, Wmat, cc


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    T = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
    M = cv2.imread(MOVING_PATH,   cv2.IMREAD_COLOR)
    assert T is not None and M is not None, f"Không đọc được ảnh:\n- {TEMPLATE_PATH}\n- {MOVING_PATH}"

    # 1) ECC align
    M_aligned, valid, Wmat, cc = ecc_align(T, M, motion=MOTION)
    print(f"[INFO] ECC cc={cc:.4f}, motion={MOTION}")

    H, W = T.shape[:2]

    # 2) Overlay toàn khung
    overlay_full = cv2.addWeighted(T, ALPHA, M_aligned, 1.0-ALPHA, 0)
    cv2.imwrite(os.path.join(OUTDIR, "01_overlay_full.png"), overlay_full)

    # 3) Crop phần giao nhau hợp lệ (minh họa)
    nz = cv2.findNonZero(valid)
    if nz is not None:
        x,y,w,h = cv2.boundingRect(nz)
        T_valid = T[y:y+h, x:x+w].copy()
        M_valid = M_aligned[y:y+h, x:x+w].copy()
        overlay_valid = cv2.addWeighted(T_valid, ALPHA, M_valid, 1.0-ALPHA, 0)
        cv2.imwrite(os.path.join(OUTDIR, "02_overlay_valid_crop.png"), overlay_valid)
    else:
        print("[WARN] Không tìm thấy vùng valid, sử dụng toàn bộ ảnh")

    # Lưu ảnh gốc và ảnh đã align
    cv2.imwrite(os.path.join(OUTDIR, "03_template.png"), T)
    cv2.imwrite(os.path.join(OUTDIR, "04_moving_aligned.png"), M_aligned)

    print("[DONE] Đã xuất ảnh demo vào:", OUTDIR)

if __name__ == "__main__":
    main()
