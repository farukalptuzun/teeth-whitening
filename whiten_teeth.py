# whiten_teeth.py
import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
import sys

mp_face_mesh = mp.solutions.face_mesh

# Mediapipe FaceMesh 468 nokta
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291]
LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308]

def imread_unicode(path: str):
    """Unicode yol desteği için güvenli okuma."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Unicode yol desteği için güvenli yazma."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False

def polygon_from_landmarks(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

def build_teeth_mask(bgr, landmarks, gum_a_cut=155, grow=3):
    """
    Ağız boşluğunu bul, dudak/diş etini ayıkla, dişler için maske üret.
    gum_a_cut: LAB a* > gum_a_cut ise diş eti (hariç tutulur)
    grow: maskeyi kaç px genişletelim (dilate)
    """
    h, w = bgr.shape[:2]
    inner_poly = polygon_from_landmarks(landmarks, LIPS_INNER, w, h)

    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [inner_poly], 255)

    # Dudak kenarından bir tık uzaklaş
    mouth_mask = cv2.erode(
        mouth_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # Renk ayrımı
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    Lc, Ac, Bc = cv2.split(lab)

    # Dudak/diş eti kırmızısı
    red1 = cv2.inRange(hsv, (0, 30, 30), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 30, 30), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # LAB a*>gum_a_cut ise gum/dudak kabul etme
    gum_mask = cv2.inRange(Ac, gum_a_cut, 255)

    not_teeth = cv2.bitwise_or(red_mask, gum_mask)

    # Diş olasılığı: (düşük S & yüksek V) veya (yüksek L & görece düşük B)
    teeth_by_hsv = cv2.bitwise_and(cv2.inRange(S, 0, 120), cv2.inRange(V, 110, 255))
    teeth_by_lab = cv2.bitwise_and(cv2.inRange(Lc, 170, 255), cv2.inRange(Bc, 0, 170))
    likely_teeth = cv2.bitwise_or(teeth_by_hsv, teeth_by_lab)

    # Ağız içi ∧ diş olasılığı ∧ ¬(kırmızı/gum)
    raw = cv2.bitwise_and(mouth_mask, cv2.bitwise_and(likely_teeth, cv2.bitwise_not(not_teeth)))

    # Çok darsa: gum hariç ağız içini baz al
    if cv2.countNonZero(raw) < 400:
        raw = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(not_teeth))

    # Temizleme
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # Biraz genişlet (gazı almak için)
    if grow > 0:
        raw = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)), 1)

    # Kenarı yumuşat
    mask = cv2.GaussianBlur(raw, (0, 0), sigmaX=3, sigmaY=3)
    return mask

def whiten_in_lab(bgr, mask, strength=0.6):
    """
    LAB uzayında:
      - L (parlaklık) artır
      - b (sarılık) azalt
    """
    # strength'i güvenli aralığa kıs
    strength = float(np.clip(strength, 0.0, 1.5))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    m = (mask.astype(np.float32) / 255.0)
    m3 = cv2.merge([m, m, m])

    l_gain = 1.0 + 0.35 * strength      # %52’ye kadar artırılabilir (1.5 ile)
    b_shift = -18.0 * strength          # sarılığı azalt

    L_new = L * (1.0 + (l_gain - 1.0) * m)
    B_new = B + (b_shift * m)

    L_new = np.clip(L_new, 0, 255)
    B_new = np.clip(B_new, 0, 255)

    lab_out = cv2.merge([L_new, A, B_new]).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # Hafif blend, aşırı beyaz patlamasını engeller
    alpha = 0.85
    blended = (out * m3 * alpha + bgr * (1 - m3 * alpha)).astype(np.uint8)
    return blended

def process_image(img_bgr, strength=0.6, gum_a_cut=155, mask_grow=3):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as fm:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return img_bgr, None
        landmarks = res.multi_face_landmarks[0].landmark

        teeth_mask = build_teeth_mask(img_bgr, landmarks, gum_a_cut=gum_a_cut, grow=mask_grow)
        out = whiten_in_lab(img_bgr, teeth_mask, strength=strength)
        return out, teeth_mask

def parse_args():
    ap = argparse.ArgumentParser(description="Diş beyazlatma (FaceMesh tabanlı)")
    # Bayraklı kullanım
    ap.add_argument("--input", help="Girdi görsel yolu")
    ap.add_argument("--output", help="Çıktı görsel yolu")
    ap.add_argument("--strength", type=float, default=0.6, help="Beyazlatma gücü (0-1.5)")
    ap.add_argument("--save-mask", action="store_true", help="Teeth mask çıktı olarak kaydet")
    ap.add_argument("--show", action="store_true", help="Önizleme penceresi aç")
    ap.add_argument("--mask-grow", type=int, default=3, help="Diş maskesini px genişlet")
    ap.add_argument("--gum-a-cut", type=int, default=155, help="LAB a* eşiği (dudak/diş eti kesimi)")
    # Pozisyonel fallback
    ap.add_argument("pos_input", nargs="?", help="(opsiyonel) input.jpg")
    ap.add_argument("pos_output", nargs="?", help="(opsiyonel) output.jpg")
    args = ap.parse_args()

    # Öncelik bayraklarda; yoksa pozisyonelleri kullan
    in_path = args.input or args.pos_input
    out_path = args.output or args.pos_output

    if not in_path or not out_path:
        ap.error("Girdi/çıktı dosyası eksik. Örnek: --input in.jpg --output out.jpg (veya) in.jpg out.jpg")

    return in_path, out_path, args

def main():
    in_path, out_path, args = parse_args()

    img = imread_unicode(in_path)
    if img is None:
        sys.exit(f"Görüntü açılamadı: {in_path}")

    out, mask = process_image(
        img_bgr=img,
        strength=args.strength,
        gum_a_cut=args.gum_a_cut,
        mask_grow=args.mask_grow
    )

    if not imwrite_unicode(out_path, out):
        sys.exit(f"Çıktı yazılamadı: {out_path}")

    if args.save_mask and mask is not None:
        mask_path = os.path.splitext(out_path)[0] + "_mask.png"
        imwrite_unicode(mask_path, mask)

    if args.show:
        cv2.imshow("original", img)
        cv2.imshow("whitened", out)
        if mask is not None:
            cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
