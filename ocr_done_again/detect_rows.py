import cv2
import numpy as np
from ocr_done_again.utils import after_lines

def rows_full(cropped, row_lines, debug=True):
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)
    # get_printed_only(binary_cropped)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # try (3,3), (5,5)
    binary_cropped = cv2.morphologyEx(binary_cropped, cv2.MORPH_OPEN, kernel)
    
    row_sum_cropped = np.sum(binary_cropped, axis=1)
    threshold_cropped = np.max(row_sum_cropped) * 0.03
    rows_detected_cropped = np.where(row_sum_cropped > threshold_cropped)[0]

    # --- Step 1: merge close lines ---
    row_lines_cropped = []
    last_y = None
    for y in rows_detected_cropped:
        if last_y is None or y - last_y > 5:
            row_lines_cropped.append(y)
        else:
            row_lines_cropped[-1] = y  # merge close rows
        last_y = y

    # --- Step 2: filter with buffer and shift ---
    buffer_y = 30
    shift_y = 10
    filtered_row_lines = []
    last_y = -999
    for y in row_lines_cropped:
        if y - last_y > buffer_y:
            filtered_row_lines.append(y + shift_y)
            last_y = y
    if debug:
        preview = cropped.copy()
        for y in filtered_row_lines:
            cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
        cv2.imwrite("output_steps/step51_manual_lines_all.jpg", preview)

    # --- Remove weak rows based on density ---
    density_threshold = 160000
    final_rows = [filtered_row_lines[0]]  # always keep first row
    for i in range(1, len(filtered_row_lines)):
        y0 = final_rows[-1]
        y1 = filtered_row_lines[i]
        density = np.sum(binary_cropped[y0:y1, :])
        if density >= density_threshold:
            final_rows.append(y1)
        else:
            # Skip this row (weak)
            pass
    # --- Draw lines on full cropped image ---
    if debug:
        preview = cropped.copy()
        for y in final_rows:
            cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
        cv2.imwrite("output_steps/step52_manual_lines_all.jpg", preview)

    # if there is only 2 lines in header
    start_row_index, _ = after_lines(row_lines, final_rows)
    if start_row_index == 2:
        final_rows.insert(2, final_rows[1]+50)

    # Remove rows near original lines
    # print(final_rows)
    top_line = row_lines[0]
    row_lines_local = [y - top_line for y in row_lines]
    # print(row_lines_local)
    TOL = 15
    filtered = [
        fy for fy in final_rows
        if not any(abs(fy - orig) <= TOL for orig in row_lines_local)
    ]
    final_rows = sorted(filtered + row_lines_local)
    # print('after operation:',final_rows)
    if debug:
        preview = cropped.copy()
        for y in final_rows:
            cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
        cv2.imwrite("output_steps/step53_manual_lines_all.jpg", preview)

    return final_rows

def rows(cropped, v_lines, fixed_col_index, row_lines):
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # try (3,3), (5,5)
    binary_cropped = cv2.morphologyEx(binary_cropped, cv2.MORPH_OPEN, kernel)
    # --- Only consider 3 columns starting from fixed_col_index ---
    if fixed_col_index + 5 <= len(v_lines):
        col_start = v_lines[fixed_col_index+1]
        col_end = v_lines[fixed_col_index + 5]  # 3 columns
        mask = np.zeros_like(binary_cropped)
        mask[:, col_start:col_end+1] = 1
        binary_for_rows = binary_cropped * mask
    else:
        binary_for_rows = binary_cropped.copy()  # fallback

    # --- Row detection using only selected columns ---
    row_sum_cropped = np.sum(binary_for_rows, axis=1)
    threshold_cropped = np.max(row_sum_cropped) * 0.03
    rows_detected_cropped = np.where(row_sum_cropped > threshold_cropped)[0]

    # --- Merge close lines ---
    row_lines_cropped = []
    last_y = None
    for y in rows_detected_cropped:
        if last_y is None or y - last_y > 5:
            row_lines_cropped.append(y)
        else:
            row_lines_cropped[-1] = y
        last_y = y

    # --- Filter with buffer and shift ---
    buffer_y = 30
    shift_y = 10
    filtered_row_lines = []
    last_y = -999
    for y in row_lines_cropped:
        if y - last_y > buffer_y:
            filtered_row_lines.append(y + shift_y)
            last_y = y
    # remove any filtered_row_lines that “touch” any of these shifted row_lines
    shifted_row_lines = [y - row_lines[0] for y in row_lines]
    shifted_row_lines = np.array(shifted_row_lines)
    filtered_row_lines = np.array(filtered_row_lines)
    region_counts = []
    region_indices = []

    for i in range(len(shifted_row_lines) - 1):
        s_start = shifted_row_lines[i]
        s_end = shifted_row_lines[i + 1]
        
        # Lines inside this region
        in_region = np.where((filtered_row_lines >= s_start) & (filtered_row_lines < s_end))[0]
        region_counts.append(len(in_region))
        region_indices.append(in_region)
    max_idx = np.argmax(region_counts)

    # Keep only the filtered lines inside this region
    filtered_cleaned = filtered_row_lines[region_indices[max_idx]]

    # Optionally, get the corresponding shifted row line boundaries
    # selected_region_start = shifted_row_lines[max_idx]
    # selected_region_end = shifted_row_lines[max_idx + 1]

    # print("Selected region:", selected_region_start, "-", selected_region_end)
    # print("Filtered lines in this region:", filtered_cleaned)

    # preview = cropped.copy()
    # for y in filtered_cleaned:
    #     cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
    # cv2.imwrite("output_steps/manual_lines_after.jpg", preview)


    # --- Remove weak rows based on density ---
    density_threshold = 200000
    final_rows = [filtered_cleaned[0]]  # always keep first row
    for i in range(1, len(filtered_cleaned)):
        y0 = final_rows[-1]
        y1 = filtered_cleaned[i]
        density = np.sum(binary_for_rows[y0:y1, :])
        if density >= density_threshold:
            final_rows.append(y1)
        else:
            # Skip this row (weak)
            pass
    # --- Draw lines on full cropped image ---
    # preview = cropped.copy()
    # for y in final_rows:
    #     cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
    # cv2.imwrite("output_steps/manual_lines_after_removing.jpg", preview)

    return final_rows