import cv2
import numpy as np
import os
from PIL import Image

image_path = '1.png'
out_dir = 'chars'
debug_dir = 'debug_masks'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")
h, w = img.shape[:2]

# Utility functions
def save_debug(name, mat):
    cv2.imwrite(os.path.join(debug_dir, name), mat)

def find_boxes(bin_img, min_area=10):
    cnts = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area or ww < 2 or hh < 2:
            continue
        boxes.append((x, y, ww, hh))
    return boxes

def save_crops(boxes, orig, prefix=''):
    saved = []
    pad = 2
    for i, (x, y, ww, hh) in enumerate(boxes, 1):
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(orig.shape[1], x+ww+pad); y2 = min(orig.shape[0], y+hh+pad)
        crop = orig[y1:y2, x1:x2]
        fname = os.path.join(out_dir, f"{prefix}char_{i:03d}.png")
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(fname)
        saved.append(fname)
    return saved

# Try multiple preprocessing variants until contours are found
median_sizes = [77, 51, 31, 15, 7]       # try large->small
threshold_methods = ['otsu', 'adaptive'] # try both
invert_options = [True, False]           # invert mask if needed
morph_kernels = [ (2,1), (2,2), (3,1), (3,3) ]

found = False
attempt = 0
results = []

for m in median_sizes:
    # avoid invalid odd/even issues for very small images
    m_blur = m if m % 2 == 1 else m+1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # local background removal
    bg = cv2.medianBlur(gray, m_blur)
    div = cv2.divide(gray, bg, scale=255)
    save_debug(f"attempt_{attempt}_div_m{m_blur}.png", div)
    # CLAHE might be helpful but try both with and without:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(div)
    save_debug(f"attempt_{attempt}_enhanced_m{m_blur}.png", enhanced)

    for thm in threshold_methods:
        if thm == 'otsu':
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 15)
        save_debug(f"attempt_{attempt}_binary_{thm}_m{m_blur}.png", binary)

        for inv in invert_options:
            bin_work = binary.copy()
            if inv:
                bin_work = cv2.bitwise_not(bin_work)
            save_debug(f"attempt_{attempt}_binary_{thm}_inv{inv}_m{m_blur}.png", bin_work)

            for kx, ky in morph_kernels:
                kernel_open = np.ones((kx, ky), np.uint8)
                # open to remove tiny noise
                cleaned = cv2.morphologyEx(bin_work, cv2.MORPH_OPEN, kernel_open, iterations=1)
                # close to connect strokes
                kernel_close = np.ones((kx, ky), np.uint8)
                final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                save_debug(f"attempt_{attempt}_final_m{m_blur}_{thm}_inv{inv}_k{kx}x{ky}.png", final)

                # optionally try small dilation to connect fragments
                dil = cv2.dilate(final, np.ones((2,2), np.uint8), iterations=1)
                save_debug(f"attempt_{attempt}_dilated_m{m_blur}_{thm}_inv{inv}_k{kx}x{ky}.png", dil)

                # search boxes on final and dilated
                boxes = find_boxes(final, min_area=20)
                boxes_dil = find_boxes(dil, min_area=20)

                # choose the larger set
                chosen = boxes if len(boxes) >= len(boxes_dil) else boxes_dil
                results.append({
                    'attempt': attempt,
                    'median': m_blur,
                    'th': thm,
                    'inv': inv,
                    'kernel': (kx, ky),
                    'boxes': len(chosen),
                    'final_img': f"attempt_{attempt}_final_m{m_blur}_{thm}_inv{inv}_k{kx}x{ky}.png"
                })

                if len(chosen) >= 1:
                    # save crops and stop (we found something)
                    boxes_sorted = sorted(chosen, key=lambda b: (b[1], b[0]))
                    prefix = f"m{m_blur}_{thm}_inv{int(inv)}_k{kx}x{ky}_"
                    saved = save_crops(boxes_sorted, img, prefix=prefix)
                    print(f"FOUND {len(saved)} boxes on attempt {attempt} -> saved to {out_dir} with prefix {prefix}")
                    found = True
                    break
                # next kernel
            if found: break
        if found: break
        attempt += 1
    if found: break

# If nothing found at all, give user suggestions and show top attempts
if not found:
    print("Не найдено контуров.")
    # show top 6 attempts with highest box counts
    results_sorted = sorted(results, key=lambda r: r['boxes'], reverse=True)
    for r in results_sorted[:6]:
        print(f"attempt {r['attempt']}: median={r['median']} th={r['th']} inv={r['inv']} kernel={r['kernel']} boxes={r['boxes']} final_image={r['final_img']}")
