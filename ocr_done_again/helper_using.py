import cv2
import numpy as np
from ocr_done_again.core import extract_data
from ocr_done_again.detect_rows import rows_full
from ocr_done_again.detect_columns import columns
from ocr_done_again.utils import crop_to_content, extract_school_code, filter_rowdected, recompute_totals, reorder_subjects, count_long_white_clusters, merge_close_lines, get_selected_box, filter_boxes_by_first_row_limits, filter_columns_from_symbol_rem, group_boxes_into_rows, normalize_text, build_flat_student_records


def NEB_OCR(img, debug=True):
    img = cv2.imread(img)
    if debug:
        cv2.imwrite("output_steps/step1_original.jpg", img)
    # --- STEP 1: Remove green ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_light_green = np.array([35, 10, 80])   # allow lower saturation, higher brightness
    upper_light_green = np.array([85, 80, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_light_green = cv2.inRange(hsv, lower_light_green, upper_light_green)
    mask_total = cv2.bitwise_or(mask_green, mask_light_green)
    img_no_green = img.copy()
    img_no_green[mask_total > 0] = [255, 255, 255]
    gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
    _, oldbinary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(oldbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_no_green = img_no_green.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 35:  # Removing if hifht is > 35
            cv2.drawContours(img_no_green, [cnt], -1,  (255, 255, 255), -1)


    # --- STEP 2: Detect first and last row and crop image---
    gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
    _, oldbinary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(oldbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary = np.zeros_like(oldbinary)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 35:  # Removing if hifht is > 35
            cv2.drawContours(binary, [cnt], -1, 255, -1)
    height, width = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))  # 5 pixels wide, 1 pixel high
    binary_expanded = cv2.dilate(binary, kernel, iterations=1)
    row_sum_filtered = np.array([count_long_white_clusters(binary_expanded[y], min_gap=200)
                                for y in range(height)])
    window_size = 10
    row_sum_multi = np.convolve(row_sum_filtered, np.ones(window_size, dtype=int), mode='same')
    threshold = np.max(row_sum_multi) * 0.18
    rows_detected_old = np.where(row_sum_multi > threshold)[0]
    rows_detected = filter_rowdected(rows_detected_old)
    continuous_thresh = 0.30
    row_lines = []
    while continuous_thresh >= 0:  # lower bound to avoid infinite loop
        row_lines = []
        for y in rows_detected:
            white_count = np.sum(binary[y] > 0)
            if white_count / width >= continuous_thresh:
                row_lines.append(y)
        row_lines = merge_close_lines(row_lines, min_dist=100)
        if len(row_lines) >= 3:  # stop if exactly 3 lines found
            top_half_count = sum(1 for y in row_lines if y < height // 2)
            bottom_half_count = sum(1 for y in row_lines if y >= height // (3/4))
            if top_half_count >= 2 and bottom_half_count >= 1:
                pass
                break  # good distribution, stop
        continuous_thresh -= 0.01   # step down threshold
    top_line = row_lines[0]
    bottom_line = row_lines[-1]
    cropped = img_no_green[top_line:bottom_line, :]
    # school
    new_top = max(top_line - 70, 0)  # ensure we don't go negative
    new_bottom = top_line
    school_cropped = img_no_green[new_top:new_bottom, :]
    gray_cropped2 = cv2.cvtColor(school_cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped2 = cv2.threshold(gray_cropped2, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # can try (3,3) or (5,5) if needed
    binary_used2 = cv2.morphologyEx(binary_cropped2, cv2.MORPH_OPEN, kernel)
    h, w = binary_used2.shape
    binary_used2_cropped = binary_used2[:, 20 : w-20]
    cropped2_content, bbox2 = crop_to_content(binary_used2_cropped)
    if cropped2_content.size != 0:
        left, top, right, bottom = bbox2
        left += 20
        right += 20
        school_cropped = school_cropped[top:bottom+1, left:right+1, :]
        boxes_sselected = [(0, 0, right-left, bottom-top)]
        ocr_text = extract_data(school_cropped, boxes_sselected)
        # print(ocr_text)
        school_code, school_name, flag = extract_school_code(ocr_text)
        # print("School_Code:", code)  # Code: 1717
        # print("School_Name:", name) 
        # print("Flag:", flag)  # True
    if debug:
        cv2.imwrite(r"output_steps/cropped2.jpg", school_cropped)
        img_no_green1 = img_no_green.copy()
        for y in rows_detected:
            cv2.line(img_no_green1, (0, y), (width, y), (0, 0, 255), 2)  # red line, thickness 2
        cv2.imwrite("output_steps/rows_detected.png", img_no_green1)
        img_no_green2 = img_no_green.copy()
        for y in row_lines:
            cv2.line(img_no_green2, (0, y), (width, y), (0, 0, 255), 2)  # red line, thickness 2
        cv2.imwrite("output_steps/rows_detected_forcrop.png", img_no_green2)


    # --- STEP 3: Detect rows and columns ---
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)
    filtered_row_lines = rows_full(cropped, row_lines)
    lines_x = columns(cropped)
    h_lines = filtered_row_lines   # horizontal lines after filtering + shift
    v_lines = lines_x              # vertical safe lines
    binary_used = binary_cropped   # binary image of cropped region
    if debug:
        preview = cropped.copy()
        for y in v_lines:
            cv2.line(preview, (y, 0), (y, cropped.shape[1]), (0, 0, 255), 1)
        cv2.imwrite("output_steps/step52_manual_vlines_all.jpg", preview)

    # --- STEP 4: Detect first column ---
    first_row_top = h_lines[0]
    first_row_bottom = h_lines[1] if len(h_lines) > 1 else binary_used.shape[0]
    first_row_region = binary_used[first_row_top:first_row_bottom, :]
    col_densities = []
    for i in range(len(v_lines)-1):
        left = v_lines[i]
        right = v_lines[i+1]
        box_region = first_row_region[:, left:right]
        density = np.sum(box_region > 0) / box_region.size
        col_densities.append(density)
    # --- Remove last column if its density is 0 ---
    if col_densities[-1] == 0:
        v_lines = v_lines[:-1]
        col_densities = col_densities[:-1]
    # --- Find fixed_col_index with density > 0.05 ---
    fixed_col_index = None
    for i, density in enumerate(col_densities):
        if density > 0.045:
            fixed_col_index = i
            break
    if fixed_col_index is None:
        fixed_col_index = 0  # fallback
    # print(fixed_col_index)

    # --- Step 5: Select boxes in rest of rows starting from fixed column ---
    boxes_selected = get_selected_box(False, row_lines, h_lines, filtered_row_lines, cropped, fixed_col_index, v_lines)
    boxes_limited = filter_boxes_by_first_row_limits(boxes_selected, cropped)
    boxes_in_original = []
    for b in boxes_limited:
        ox1, oy1, ox2, oy2 = b["original"]
        ax1, ay1, ax2, ay2 = b["adjusted"]
        boxes_in_original.append({
            "original": (ox1, oy1 + top_line, ox2, oy2 + top_line),
            "adjusted": (ax1, ay1 + top_line, ax2, ay2 + top_line)
        })
    if debug:
        img_original_boxes = img.copy()
        img_adjusted_boxes = img.copy()
        for b in boxes_in_original:
            # original box (blue)
            x1, y1, x2, y2 = b["original"]
            cv2.rectangle(img_original_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue

            # adjusted box (green)
            x1, y1, x2, y2 = b["adjusted"]
            cv2.rectangle(img_adjusted_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        cv2.imwrite("output_steps/original_boxes.jpg", img_original_boxes)
        cv2.imwrite("output_steps/adjusted_boxes.jpg", img_adjusted_boxes)

    adjusted_boxes = [b["adjusted"] for b in boxes_in_original]

    # --- Step 6: Extracting Text from images ---
    result_text = extract_data(img, adjusted_boxes)
    # print(result_text)
    boxes_filtered, texts_filtered = filter_columns_from_symbol_rem(boxes_limited, result_text)
    rows_text = group_boxes_into_rows(boxes_filtered, texts_filtered)
    normalized_rows  = normalize_text(rows_text)
    data = build_flat_student_records(normalized_rows)


    # --- Step 7 :Removing the subject took from the writen rows (Mistaken) ---
    cleaned_data = reorder_subjects(data)
    # cleaned_data = []
    # for stu in data1:
    #     count = 0
    #     for k, v in stu.items():
    #         if k.startswith(('CODE')):
    #             count += 1
    #     if count <= 7:
    #         cleaned_data.append(stu)
    cleaned_data = recompute_totals(cleaned_data)
    # for i, stu in enumerate(cleaned_data, 1):
    #     row = " | ".join(f"{k}={v}" for k, v in stu.items())
    #     print(f"Student {i}: {row}\n")


    # --- Step 8  :Updating to databases---

    # for i, stu in enumerate(data, 1):
    #     row = " | ".join(f"{k}={v}" for k, v in stu.items())
    #     print(f"Student {i}: {row}")
    #     print("")
    for row in cleaned_data:
        row['School_Code'] = school_code if school_code else None
        row['School_Name'] = school_name if school_name else None
    return cleaned_data