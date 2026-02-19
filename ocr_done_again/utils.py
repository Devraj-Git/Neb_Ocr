
import cv2, difflib, re
import numpy as np
from ocr_done_again.detect_columns import merged_col_operation
from itertools import combinations

def get_printed_only(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    output = np.zeros_like(binary)

    for i, stat in enumerate(stats):
        if i == 0:
            continue  # background

        x, y, w, h, area = stat

        # Filter based on height (tune min_h and max_h)
        min_h = 10
        max_h = 30

        if min_h <= h <= max_h:
            output[labels == i] = 255  # keep printed text

    # Invert back if needed
    result = cv2.bitwise_not(output)

    # cv2.imwrite("printed_only.png", result)

def after_lines(row_lines, h_lines, ):
    # row1_in_crop = row_lines[1] -  row_lines[0]
    # start_row_index = 0
    # for i in range(len(h_lines) - 1): # Selecting boxes after lines
    #     if h_lines[i] >= row1_in_crop:
    #         start_row_index = i
    #         if start_row_index > 0:
    #             return start_row_index, start_row_index-1
    #         else:
    return 0, None

def count_long_white_clusters(row, min_gap=5, max_black_ratio=0.7):
    """
    Count only white clusters in a row that are at least `min_gap` pixels long.
    row: 1D binary array (0 or 255)
    """
    width = len(row)
    black_count = np.sum(row == 0)
    if black_count / width > max_black_ratio:
        return 0
    count = 0
    current_len = 0
    for pixel in row:
        if pixel > 0:  # white pixel
            current_len += 1
        else:  # black pixel
            if current_len >= min_gap:
                count += current_len  # count only long enough cluster
            current_len = 0
    # check at the end of row
    if current_len >= min_gap:
        count += current_len
    return count        

def merge_close_lines(lines, min_dist=15):
    """Keep only one line per cluster if lines are closer than min_dist"""
    if not lines:
        return []
    lines = sorted(lines)
    merged = [lines[0]]
    for y in lines[1:]:
        if y - merged[-1] > min_dist:
            merged.append(y)
        # else skip (too close to previous)
    return merged


def clean_box(boxes_selected, binary_used, h_lines, v_lines):
    line_margin = 2000 # how many pixels around the line to consider "touching"

    boxes_clean = []

    for (x1, y1, x2, y2) in boxes_selected:
        roi = binary_used[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape

        # check horizontal lines
        touches_h_line = False
        for h in h_lines:
            if y1 - line_margin <= h <= y2 + line_margin:  # only lines near the box
                roi_y = h - y1
                if 0 <= roi_y < roi_h and np.any(roi[roi_y, :]):
                    touches_h_line = True
                    break

        # check vertical lines
        touches_v_line = False
        for v in v_lines:
            if x1 - line_margin <= v <= x2 + line_margin:
                roi_x = v - x1
                if 0 <= roi_x < roi_w and np.any(roi[:, roi_x]):
                    touches_v_line = True
                    break

        if touches_h_line or touches_v_line:
            continue
        boxes_clean.append((x1, y1, x2, y2))
    return boxes_clean


def count_first_row_boxes(boxes, y_tolerance=5):
    """
    Count the number of boxes in the first row.
    
    Args:
        boxes: list of tuples (x1, y1, x2, y2)
        y_tolerance: maximum vertical difference to consider the same row
        
    Returns:
        count: int
        first_row_boxes: list of boxes in the first row
    """
    if not boxes:
        return 0, []

    # Find the minimum y1 value (top of the first row)
    min_y1 = min(box[1] for box in boxes)

    # Select boxes that are within y_tolerance of the top
    first_row_boxes = [box for box in boxes if abs(box[1] - min_y1) <= y_tolerance]

    return len(first_row_boxes)

def count_unique_vertical_lines(boxes, y_tolerance=5):
    """
    Get unique vertical lines in the first row:
    - left_lines: left edge of a box that is not any box's right edge
    - right_lines: right edge of a box that is not any box's left edge

    Args:
        boxes: list of tuples (x1, y1, x2, y2)
        y_tolerance: maximum vertical difference to consider same row

    Returns:
        left_lines: list of x-coordinates
        right_lines: list of x-coordinates
    """
    if not boxes:
        return [], []

    # Find top y-coordinate of first row
    min_y1 = min(box[1] for box in boxes)

    # Filter boxes in the first row
    first_row_boxes = [box for box in boxes if abs(box[1] - min_y1) <= y_tolerance]

    # Get left and right edges
    left_edges = [box[0] for box in first_row_boxes]
    right_edges = [box[2] for box in first_row_boxes]

    # Unique left: left edge not any box's right edge
    unique_left = [x for x in left_edges if x not in right_edges]

    # Unique right: right edge not any box's left edge
    unique_right = [x for x in right_edges if x not in left_edges]

    return unique_left, unique_right


def enforce_row_box_count(boxes, y_tolerance=5, total_columns=24):
    """
    For each row, ensure the number of boxes matches the expected count based on unique left lines.
    Remove extra boxes from the right if necessary.

    Args:
        boxes: list of tuples (x1, y1, x2, y2)
        y_tolerance: maximum vertical difference to consider same row
        total_columns: total number of columns expected

    Returns:
        filtered_boxes: list of boxes after enforcing box count per row
    """
    if not boxes:
        return []

    filtered_boxes = []

    # Sort boxes by top-left corner (y, then x)
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))

    while boxes_sorted:
        # Start a new row with the topmost box
        row_y1 = boxes_sorted[0][1]

        # Select all boxes in this row
        row_boxes = [b for b in boxes_sorted if abs(b[1] - row_y1) <= y_tolerance]

        # Remove them from boxes_sorted
        boxes_sorted = [b for b in boxes_sorted if abs(b[1] - row_y1) > y_tolerance]

        # Compute unique left lines
        left_edges = [b[0] for b in row_boxes]
        right_edges = [b[2] for b in row_boxes]
        unique_left = [x for x in left_edges if x not in right_edges]

        expected_count = total_columns - len(unique_left)

        # If there are too many boxes, remove from the right
        if len(row_boxes) > expected_count:
            row_boxes = sorted(row_boxes, key=lambda b: b[0])[:expected_count]

        filtered_boxes.extend(row_boxes)

    return filtered_boxes


def filter_boxes_by_first_row_limits(boxes, img, y_tolerance=5, tolerance=2):
    """
    Filter boxes based on first row:
    - If a box starts after the 3rd column, it must end before or at the 3rd-last column.
    - Otherwise, keep the box.

    Args:
        boxes: list of tuples (x1, y1, x2, y2)
        y_tolerance: vertical tolerance to group rows

    Returns:
        filtered_boxes: list of boxes satisfying the condition
    """
    image_width = img.shape[1]
    second = False
    if not boxes:
        return []

    filtered_boxes = []

    boxes_sorted = sorted(boxes, key=lambda b: (b["adjusted"][1], b["adjusted"][0]))

    first_row_y1 = boxes_sorted[0]["adjusted"][1]
    first_row_boxes = [b for b in boxes_sorted if abs(b["adjusted"][1] - first_row_y1) <= y_tolerance]
    first_row_boxes = sorted(first_row_boxes, key=lambda b: b["adjusted"][0])
    

    if len(first_row_boxes) >= 3:
        start_limit = first_row_boxes[2]["adjusted"][0]
    else:
        start_limit = first_row_boxes[0]["adjusted"][0]
    # end_limit = first_row_boxes[-2][0]
    last_box_x = first_row_boxes[-1]["adjusted"][0]
    if last_box_x > image_width * 0.75:
        # Use first row's second-to-last box
        # end_limit = first_row_boxes[-2][0]
        if len(first_row_boxes) >= 2:
            end_limit = first_row_boxes[-2]["adjusted"][0]
        else:
            # fallback option, e.g. use last one
            end_limit = first_row_boxes[-1]["adjusted"][0] if first_row_boxes else None
    else:
        second = True
        second_row_y1 = None
        for b in boxes_sorted:
            if abs(b["adjusted"][1] - first_row_y1) > y_tolerance:
                second_row_y1 = b["adjusted"][1]
                break
        # Collect second row boxes
        second_row_boxes = []
        if second_row_y1 is not None:
            second_row_boxes = [b for b in boxes_sorted if abs(b["adjusted"][1] - second_row_y1) <= y_tolerance]
            second_row_boxes = sorted(second_row_boxes, key=lambda b: b["adjusted"][0])
        if second_row_boxes:
            end_limit = second_row_boxes[-2]["adjusted"][0]
        else:
            end_limit = image_width  # fallback
    
    while boxes_sorted:
        row_y1 = boxes_sorted[0]["adjusted"][1]
        row_boxes = [b for b in boxes_sorted if abs(b["adjusted"][1] - row_y1) <= y_tolerance]
        boxes_sorted = [b for b in boxes_sorted if abs(b["adjusted"][1] - row_y1) > y_tolerance]
        row_boxes.sort(key=lambda b: b["adjusted"][0])
        
        for i in range(len(row_boxes) - 1):
            curr = row_boxes[i]
            nxt = row_boxes[i + 1]

            x1_c, y1_c, x2_c, y2_c = curr["adjusted"]
            x1_n, y1_n, x2_n, y2_n = nxt["adjusted"]

            if x2_c > x1_n:  # overlap detected
                midpoint = int((x2_c + x1_n) / 2)

                # Adjust both boxes
                curr["adjusted"] = (x1_c, y1_c, midpoint, y2_c)
                nxt["adjusted"] = (midpoint, y1_n, x2_n, y2_n)

        row_boxes_filtered = []
        skip = False
        for b in row_boxes:
            x1, x2 = b["adjusted"][0], b["adjusted"][2]
            keep = True
            if x1 >= start_limit:
                if x2 <= end_limit + tolerance or second or skip:
                    row_boxes_filtered.append(b)
                else:
                    keep=False
            else:
                row_boxes_filtered.append(b)
                skip = True

        filtered_boxes.extend(row_boxes_filtered)

    return filtered_boxes


def crop_to_content(roi):
    """
    Crops a binary image to the bounding box of non-zero pixels.
    Returns an empty array if no content is found.
    """
    ys, xs = np.where(roi > 0)
    if ys.size == 0 or xs.size == 0:
        return np.array([]), None  # no content
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    return roi[top:bottom+1, left:right+1], (left, top, right, bottom)


def crop_to_content_new(roi, min_height=8):
    """
    Crops a binary ROI to the bounding box of all non-zero pixels first,
    then removes small blobs whose height is less than min_height.
    Returns an empty array if no valid content remains.
    """
    # Step 1: Find bounding box of all non-zero pixels
    ys, xs = np.where(roi > 0)
    if ys.size == 0 or xs.size == 0:
        return np.array([]), None  # nothing to keep

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    
    # Crop the ROI tightly
    roi_cropped = roi[top:bottom+1, left:right+1]

    # Step 2: Remove small blobs inside the cropped ROI
    contours, _ = cv2.findContours(roi_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(roi_cropped)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_height:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Step 3: Return final cropped content
    ys2, xs2 = np.where(mask > 0)
    if ys2.size == 0 or xs2.size == 0:
        return np.array([]), None  # nothing left after removing small blobs

    top2, bottom2 = ys2.min(), ys2.max()
    left2, right2 = xs2.min(), xs2.max()
    return mask[top2:bottom2+1, left2:right2+1], (left2, top2, right2, bottom2)


def get_selected_box(case, row_lines, h_lines, filtered_row_lines, cropped, fixed_col_index, v_lines, min_density = 0.0025, debug=False):
    """
        Case=True :- for uniform column
        Case=False :- for row based column
    """
    if debug:
        preview = cropped.copy()
        for y in filtered_row_lines:
            cv2.line(preview, (0, y), (cropped.shape[1], y), (0, 0, 255), 1)
        for x in v_lines:
            cv2.line(preview, (x, 0), (x, cropped.shape[0]), (0, 0, 255), 1)
        cv2.imwrite("output_steps/both_lines.jpg", preview)

    boxes_selected = []
    # start_row_index, _ = after_lines(row_lines, h_lines) # Selecting boxes after lines
    row_vlines = None
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # try (3,3), (5,5)
    binary_used = cv2.morphologyEx(binary_cropped, cv2.MORPH_OPEN, kernel)

    img_height, img_width = cropped.shape[:2]
    for i in range(len(h_lines)-1):  # start from 1st row
        y1, y2 = h_lines[i], h_lines[i+1]
        if case:
            for j in range(fixed_col_index, len(v_lines)-1):  # start from fixed column
                x1, x2 = v_lines[j], v_lines[j+1]
                roi = binary_used[y1:y2, x1:x2]
                roi_cropped, offsets = crop_to_content(roi)
                if roi_cropped.size == 0:
                    continue
                # plt.imshow(roi_cropped, cmap="gray")
                # plt.title(f"ROI ({i},{j})")
                # plt.axis("off")
                # plt.show()
                density = np.sum(roi_cropped > 0) / roi.size
                if y2 >= img_height:
                    continue
                if density >= min_density and np.sum(roi_cropped > 0) > 100:
                    boxes_selected.append((x1, y1, x2, y2))
        else:
            if not row_vlines:
                # after_line_index, before_line_index = after_lines(row_lines, filtered_row_lines)
                row_vlines = merged_col_operation(cropped, 0, None, filtered_row_lines, row_lines)
                if debug:
                    preview = cropped.copy()
                    for y in row_vlines:
                        cv2.line(preview, (y, 0), (y, cropped.shape[1]), (0, 0, 255), 1)
                    cv2.imwrite("output_steps/step60_merged_col_operation.jpg", preview)
                    preview = cropped.copy()
                    for y in h_lines:
                        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 0, 255), 1)
                height, width = cropped.shape[:2]
                rows_of_interest = sorted(row_vlines.keys())  # these are the row indices in filtered_row_lines
                for ij, row_idx in enumerate(rows_of_interest):
                    lines_x = row_vlines[row_idx]

                    # get row boundaries using filtered_row_lines or h_lines slice
                    top = h_lines[row_idx]
                    if (row_idx + 1) < len(h_lines):
                        bottom = h_lines[row_idx + 1]
                    else:
                        bottom = height  # last row
                    if debug:
                        for x in lines_x:
                            cv2.line(preview, (x, top), (x, bottom), (0, 0, 255), 1)
                if debug:
                    cv2.imwrite("output_steps/step2_both_lines.jpg", preview)

            v_lines_here = row_vlines.get(i, [])
            for j in range(len(v_lines_here)-1):  # start from fixed column
                x1, x2 = v_lines_here[j], v_lines_here[j+1]
                roi = binary_used[y1:y2, x1:x2]
                roi_cropped, offsets = crop_to_content(roi)
                if roi_cropped.size == 0:
                    continue
                # plt.imshow(roi_cropped, cmap="gray")
                # plt.title(f"ROI ({i},{j})")
                # plt.axis("off")
                # plt.show()
                density = np.sum(roi_cropped > 0) / roi.size
                if y2 >= img_height:
                    continue
                if density >= min_density and np.sum(roi_cropped > 0) > 100:
                    left, top, right, bottom = offsets
                    gx1 = x1 + left
                    gx2 = x1 + right
                    gy1 = y1 
                    gy2 = y2
                    pad = 20
                    gx1 = max(0, gx1 - pad)
                    gx2 = min(img_width, gx2 + pad)
                    if roi_cropped.shape[1] > 900:  # check width
                        cropped_with_pad = binary_used[gy1:gy2, gx1:gx2]
                        fixed_width = 500   # <-- adjust as needed
                        if cropped_with_pad.shape[1] > fixed_width:
                            # Re-crop horizontally around the densest part
                            col_sum = np.sum(cropped_with_pad > 0, axis=0)
                            center = np.argmax(col_sum)  # column with max content
                            half_w = fixed_width // 2
                            cx1 = max(0, center - half_w)
                            cx2 = min(cropped_with_pad.shape[1], center + half_w)
                            cropped_with_pad = cropped_with_pad[:, cx1:cx2]
                            # Update global coords
                            gx1 = gx1 + cx1
                            gx2 = gx1 + cropped_with_pad.shape[1]

                        # plt.imshow(cropped_with_pad, cmap="gray")
                        # plt.title(f"Global ROI ({i},{j}) gx1={gx1}, gy1={gy1}, gx2={gx2}, gy2={gy2}")
                        # plt.axis("off")
                        # plt.show()

                        boxes_selected.append({
                            "original": (x1, y1, x2, y2),
                            "adjusted": (gx1, gy1, gx2, gy2)
                        })
                    else:
                        boxes_selected.append({
                            "original": (x1, y1, x2, y2),
                            "adjusted": (gx1, gy1, gx2, gy2)
                        })
    return boxes_selected


def group_boxes_into_rows(boxes, texts, tol=10):
    """
    Group boxes and corresponding OCR texts into rows.
    
    Args:
        boxes: list of dicts with keys {"original": (x1,y1,x2,y2), "adjusted": (x1,y1,x2,y2)}
        texts: list of OCR results in the same order
        tol: vertical tolerance to group boxes into the same row
        use_adjusted: if True, grouping is based on adjusted coordinates; else original

    Returns:
        rows: list of rows, each row is a list of tuples:
              (text, adjusted_box, original_box)
    """
    # Attach text to box
    items = [(b["adjusted"][0], b["adjusted"][1], b["adjusted"][2], b["adjusted"][3], txt, b["original"], b["adjusted"]) for b, txt in zip(boxes, texts)]
    
    # Sort by y first, then by x
    items.sort(key=lambda b: (b[1], b[0]))
    
    rows = []
    current_row = []
    last_y = None
    
    for x1, y1, x2, y2, txt, orig_box, adj_box in items:
        if last_y is None:
            last_y = y1
            current_row.append((txt, adj_box, orig_box))
        elif abs(y1 - last_y) <= tol:  # same row
            current_row.append((txt, adj_box, orig_box))
        else:  # new row
            # sort row by x before appending
            current_row.sort(key=lambda b: b[1][0])
            # rows.append([t for t, _ in current_row])
            rows.append(current_row)
            
            current_row = [(txt, adj_box, orig_box)]
            last_y = y1
    
    # Add last row
    if current_row:
        current_row.sort(key=lambda b: b[1][0])
        # rows.append([t for t, _ in current_row])
        rows.append(current_row)
    
    return rows

def clean_ocr_parts(parts):
    cleaned = []
    for i, p in enumerate(parts):
        p = str(p)
        # Only fix month/day (parts after year)
        if i > 0:
            if len(p) == 3 and (p.startswith("7") or p.endswith("7")):
                # remove the 7 at start or end
                p = p[1:] if p.startswith("7") else p[:-1]
        # Ensure 2 digits
        p = p.zfill(2)
        cleaned.append(p)
    return cleaned

def normalize_after_key(key, value, row):
    def format_date_exact(s):
        s = str(s).strip()
        # Case 1: already in yyyy/mm/dd
        if len(s) == 10 and s[4] == "/" and s[7] == "/":
            y, m, d = s.split("/")
            if y.isdigit() and m.isdigit() and d.isdigit() and (y.startswith("19") or y.startswith("20")):
                return s
            return ""
        # Case 2: raw yyyymmdd
        if len(s) == 8 and s.isdigit():
            y, m, d = s[:4], s[4:6], s[6:]
            if y.startswith("19") or y.startswith("20"):
                return f"{y}/{m}/{d}"
            return ""
        return ""
    if key == 'DOB':
        value = re.sub(r'[^0-9/,]', '', value)
        match = re.search(r'(\d{4}/\d{2}/\d{2})', value)
        if match:
            date_val = match.group(1)
            return key, date_val
        parts = []
        current_idx = None
        for i, (txt, _) in enumerate(row):
            if value in txt:
                current_idx = i
                break
        if current_idx is not None:
            j = current_idx
            while j < current_idx+3 and j < len(row):
                nxt_txt, _ = row[j]
                nxt_val = re.sub(r'[^0-9]', '', nxt_txt)
                if nxt_val.isdigit():
                    digits = nxt_val
                    if j == current_idx:
                        if len(digits) == 5:
                            digits = digits[:-1]  # remove last one
                    elif j == current_idx + 1:
                        if len(digits) == 4:
                            digits = digits[1:3]  # take only middle 2 digits
                        elif len(digits) == 3:
                            if digits[0] in ('7', '1') and len(digits) > 2:
                                digits = digits[1:]  # remove first digit
                            elif digits[-1] in ('7', '1') and len(digits) > 2:
                                digits = digits[:-1]  # remove last digit
                            if len(digits) < 2:
                                digits = digits.zfill(2)
                    elif j == current_idx + 2:
                        if len(digits) == 3 and not '/' in nxt_txt:
                            digits = digits[1:]  # remove first one to make it 2
                        if len(digits) > 2:
                            digits = digits[:2]
                    parts.append(digits)
                j += 1
        parts = clean_ocr_parts(parts)
        formatted = ""
        if len(parts) == 3:
            formatted = f"{parts[0]}/{parts[1].zfill(2)}/{parts[2].zfill(2)}"
        elif len(parts) == 2:
            # year + month/day (but incomplete)
            formatted = f"{parts[0]}/{parts[1].zfill(2)}"
        elif len(parts) == 1:
            formatted = parts[0]

        if re.fullmatch(r'\d{4}/\d{2}/\d{2}', formatted):
            return key, formatted
        else:
            return key, format_date_exact(formatted)
    elif key.startswith(('TH', 'PR', 'TOT')) and key != 'TOTAL':
        if value == ":":
            value = "A"
        elif str(value).isalpha():
            key = "CODE"
        else:
            remove_list = ["~", " ", ":", "%"]
            if key.startswith("TOT"):
                remove_list.append("*")
            for ch in remove_list:
                value = value.replace(ch, "")
            m = re.match(r'^(\d{2})(.+)', str(value))
            if m:
                if key.startswith('TOT'):
                    value = m.group(1)
                else:
                    value = m.group(1) + "*"   # keep first 2 digits, then one *
        return key, value
    elif key == 'TOTAL':
        m = re.search(r"\d+", str(value))
        if m:
            value = m.group(0)   # keep only integer part
        else:
            value = 0  # or "0" if you prefer default
        return key, value
    elif key == 'CODE':
        # Allow letters, spaces, dot, and ampersand
        value = re.sub(r"[^A-Za-z\s.&]", "", str(value)).strip()
        return key, value
    
    elif key == 'NAME OF THE STUDENT':
        value = re.sub(r"[^A-Z\s]", "", str(value)).strip()
        return key, value
    
    elif key == 'SYMBOL':
        clean_val  = re.sub(r"[^0-9\s]", "", str(value)).strip()
        parts = clean_val.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            if len(parts[0]) <= 2 and len(parts[1]) > len(parts[0]):
                clean_val = parts[1]
        return key, clean_val
    return key, value


def build_flat_student_records(rows):
    header1 = rows[0]  # main headers
    header2 = rows[1]  # extra headers like DOB, AMOUNT
    students = []

    def x_center(box):
        if not box:
            return None
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2

    # helper: snap a row to a set of headers
    def snap_row(row, headers):
        mapping = []
        used_cells = set()  # keep track of already matched cell indices

        for h_text, h_adj_box, h_orig_box in headers:
            if not h_text.strip():
                continue
            hx = x_center(h_orig_box)
            if hx is None:
                continue
            best_val = ""
            for idx, (txt, adj_box, orig_box) in enumerate(row):
                if idx in used_cells:
                    continue  # skip already matched cells
                if not orig_box:
                    continue
                cx1, cy1, cx2, cy2 = orig_box
                if h_text.strip().upper() == "DOB":
                    if hx >= cx1 - 100 and hx <= cx2 + 100:
                        best_val = txt
                        used_cells.add(idx)
                        break
                else:
                    if hx >= cx1 and hx <= cx2:
                        best_val = txt
                        used_cells.add(idx)  # mark this cell as used
                        break
            h,v = normalize_after_key(h_text, best_val, row)
            mapping.append((h, v, h_orig_box))

        unmatched = [cell for idx, cell in enumerate(row) if idx not in used_cells]

        return mapping, unmatched

    # iterate in pairs: odd+even rows per student
    for i in range(2, len(rows), 2):
        odd_row = rows[i]
        even_row = rows[i+1] if i+1 < len(rows) else []
        record = {}

        # snap main row
        snapped, _ = snap_row(odd_row, header1)

        # track subject blocks
        subj_count = 0
        for h_text, val, h_box in snapped:
            h_text_upper = h_text.strip().upper()
            if h_text_upper in ["CODE", "TH", "PR", "TOT"]:  # repeated block
                subj_count += 1 if h_text_upper == "CODE" else 0
                if val == "A" and h_text_upper == "CODE":
                    if subj_count > 1:
                        count = subj_count - 1
                    else:
                        count = subj_count
                    key = f"TH{count}"
                else:
                    key = f"{h_text_upper}{subj_count}"
            else:
                key = h_text
            record[key] = val

        # snap extra row (DOB, AMOUNT, etc.)
        snapped_extra, unmatched_cells = snap_row(even_row, header2)
        for h_text, val, h_box in snapped_extra:
            h_text_upper = h_text.strip().upper()
            if h_text_upper in ["CODE", "TH", "PR", "TOT"]:  # repeated block
                subj_count += 1 if h_text_upper == "CODE" else 0
                if val == "A" and h_text_upper == "CODE":
                    if subj_count > 1:
                        count = subj_count - 1
                    else:
                        count = subj_count
                    key = f"TH{count}"
                else:
                    key = f"{h_text_upper}{subj_count}"
            else:
                key = h_text
            if key not in record:
                record[key] = val

        # --- fallback: find remaining values in even_row not matched ---
        # unmatched_cells = [cell for cell in even_row if all(cell[0] != v for _, v, _ in snapped_extra)]
        if unmatched_cells:
            snapped_fallback, _ = snap_row(unmatched_cells, header1)
            for h_text, val, h_box in snapped_fallback:
                if (h_text not in record or not str(record[h_text]).strip()) and val != '':
                    h_text_upper = h_text.strip().upper()
                    if h_text_upper in ["CODE", "TH", "PR", "TOT"]:  # repeated block
                        subj_count += 1 if h_text_upper == "CODE" else 0
                        if val == "A" and h_text_upper == "CODE":
                            if subj_count > 1:
                                count = subj_count - 1
                            else:
                                count = subj_count
                            key = f"TH{count}"
                        else:
                            key = f"{h_text_upper}{subj_count}"
                    else:
                        key = h_text
                    record[key] = val

        students.append(record)

    return students



def filter_columns_from_symbol_rem(boxes, texts, keyword1="SYMBOL", keyword2="REM", similarity=0.7):
    """
    Find the column where header ≈ keyword1 and drop all boxes left of it.
    Find the column where header ≈ keyword2 and drop all boxes right of it.
    """
    # Attach text to box
    items = [(b["adjusted"][0], b["adjusted"][1], b["adjusted"][2], b["adjusted"][3], txt) for b, txt in zip(boxes, texts)]
    
    # Take only first row candidates (smallest y)
    y_values = sorted({y1 for (_, y1, _, _, _) in items})
    rows = []

    for row_y in y_values[:2]:  # only first 2 rows
        row = [(x1, x2, txt) for (x1, y1, x2, _, txt) in items if abs(y1 - row_y) < 10]
        row.sort(key=lambda b: b[0])  # sort left-to-right
        rows.append(row)

    # --- Step 2: scoring function ---
    def get_best_scores(row, keyword1, keyword2):
        best_x, best_x2 = None, None
        best_score, best_score2 = 0, 0
        for x1, x2, txt in row:
            score = difflib.SequenceMatcher(None, txt.upper(), keyword1.upper()).ratio()
            score2 = difflib.SequenceMatcher(None, txt.upper(), keyword2.upper()).ratio()
            if score > best_score:
                best_score = score
                best_x = x1
            if score2 > best_score2:
                best_score2 = score2
                best_x2 = x2
        return best_x, best_x2, best_score, best_score2

    # --- Step 3: check both rows and keep the overall best ---
    overall_best_x, overall_best_x2 = None, None
    overall_best_score, overall_best_score2 = 0, 0

    for row in rows:
        best_x, best_x2, best_score, best_score2 = get_best_scores(row, keyword1, keyword2)
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_x = best_x
        if best_score2 > overall_best_score2:
            overall_best_score2 = best_score2
            overall_best_x2 = best_x2
    
    if overall_best_score < similarity and overall_best_score2 < similarity:
        # print("⚠️ No good match found for SYMBOL and REM")
        return boxes, texts
    
    # print(f"SYMBOL column detected at x={overall_best_x}, similarity={overall_best_score:.2f}")
    # print(f"REM column detected at x={overall_best_x2}, similarity={overall_best_score2:.2f}")
    
    # Keep only boxes with x1 >= SYMBOL column start
    filtered_boxes = []
    filtered_texts = []
    for b, txt in zip(boxes, texts):
        x1, y1, x2, y2 = b["adjusted"]
        if x2-30 >= overall_best_x and x1 <= overall_best_x2:
            filtered_boxes.append(b)
            filtered_texts.append(txt)
    
    return filtered_boxes, filtered_texts



def normalize_text(rows_text, status_threshold=0.6, header_threshold=0.4):
    """
    Normalize OCR text rows:
      - Fix headers using expected_headers and difflib
      - Fix status strings (PASS, FAIL1..FAIL7) using difflib
    """
    expected_headers = [
        'SYMBOL', 'REG.NO.', 'NAME OF THE STUDENT', 'DOB', '& DOB',
        'CODE','TH','PR','TOT',
        'TOTAL','REM'
    ]
    VALID_STATUSES = ["PASS"] + [f"FAIL{i}" for i in range(1, 8)]
    OCR_FIX = str.maketrans({"I":"1", "l":"1", "O":"0", "o":"0", " ":""})
    WORD_FIXES = {
        "SUBJ": "CODE",
        "& DOB": "DOB",
        "@ 00B": "DOB",
        "00B": "DOB",
    }
    HEADER_FIXES_AFTER_TOTAL = {"TH", "PR", "SUBJ", "CODE", "TOT"}
    def normalize_status(cell, row):
        cell = cell.strip()
        for old, new in WORD_FIXES.items():
            cell = cell.replace(old, new)
        match = re.fullmatch(r"([0-9.]+)(\*)([\s\*]*)", cell)
        if match:
            digits, star, _ = match.groups()
            cell = digits.replace(".", "") + star
        # Extract first valid status pattern.
        txt = cell.upper()
        match = re.search(r"(PASS|FAIL[1-7])", txt)
        if match:
            return match.group(1)
        else:
            txt = cell.upper().translate(OCR_FIX)
            max_ratio = 0
            for status in VALID_STATUSES:
                ratio = difflib.SequenceMatcher(None, txt, status).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
            if max_ratio >= status_threshold:
                values = [c for c, _, _ in row]
                fail_count = 0
                for i in range(0, len(values), 3):
                    group = values[i:i+3]
                    if any(re.search(r"\d+\*", str(v)) for v in group):
                        fail_count += 1
                if fail_count == 0:
                    return "PASS"
                else:
                    fail_num = min(fail_count, 7)
                    return f"FAIL{fail_num}"
        return cell  # no change

    normalized_rows = []

    for row_idx, row in enumerate(rows_text):
        new_row = []
        for col_idx, (cell, adj_box, orig_box) in enumerate(row):
            txt = cell.strip()
            for old, new in WORD_FIXES.items():
                txt = txt.replace(old, new)

            # --- Header normalization ---
            if expected_headers and row_idx < 2:  # assuming first 1-2 rows are headers
                if row_idx==1:
                    txt = re.sub(r"[^A-Za-z. ]+", "", txt)
                if row_idx==2:
                    txt = re.sub(r"[^A-Za-z@0. ]+", "", txt)
                match = difflib.get_close_matches(txt.upper(), expected_headers, n=1, cutoff=header_threshold)
                if match:
                    txt = match[0]
                else:
                    continue
                if txt.upper() == "TOTAL" and col_idx + 1 < len(row):
                    next_cell = row[col_idx + 1][0].strip().upper()
                    if next_cell in HEADER_FIXES_AFTER_TOTAL:
                        txt = "TOT"

            # --- Status normalization ---
            txt = normalize_status(txt, row)

            new_row.append((txt, adj_box, orig_box))
        normalized_rows.append(new_row)

    return normalized_rows


def reorder_subjects(ocr_results, max_subjects=7):
    """
    Reorder subjects in OCR results so that the first non-empty CODE becomes CODE1, TH1, PR1, TOT1, etc.
    Works for a single dict or a list of dicts.
    """
    # Ensure we're always working with a list
    if isinstance(ocr_results, dict):
        ocr_results = [ocr_results]

    for ocr_result in ocr_results:
        # Collect all subjects
        subjects = []
        for key, value in ocr_result.items():
                if key.startswith("CODE") and value:  # only if CODE exists and is not empty
                    i = key.replace("CODE", "")  # extract the subject number
                    subjects.append({
                        "CODE": value,
                        "TH": ocr_result.get(f"TH{i}") or None,
                        "PR": ocr_result.get(f"PR{i}") or None,
                        "TOT": ocr_result.get(f"TOT{i}") or None
                    })

        # --- Clear old CODE/TH/PR/TOT ---
        keys_to_remove = [k for k in list(ocr_result.keys()) if k.startswith(("CODE", "TH", "PR", "TOT")) and k != "TOTAL" ]
        for k in keys_to_remove:
            ocr_result.pop(k, None)

        # --- Find max subjects you want (example: 7) ---
        max_subjects = 7  

        # --- Reassign with padding ---
        for idx in range(1, max_subjects + 1):
            if idx <= len(subjects):
                subj = subjects[idx - 1]
                ocr_result[f"CODE{idx}"] = subj["CODE"]
                ocr_result[f"TH{idx}"]   = subj["TH"]
                ocr_result[f"PR{idx}"]   = subj["PR"]
                ocr_result[f"TOT{idx}"]  = subj["TOT"]
            else:
                ocr_result[f"CODE{idx}"] = None
                ocr_result[f"TH{idx}"]   = None
                ocr_result[f"PR{idx}"]   = None
                ocr_result[f"TOT{idx}"]  = None

    return ocr_results


def recompute_totals(ocr_results, max_subjects=7):
    """
    Recompute TOT = TH + PR for each subject dynamically.
    Handles '*' or other non-digit characters in TH/PR.
    Updates TOT only if it's missing or different.
    """
    if isinstance(ocr_results, dict):
        ocr_results = [ocr_results]

    for ocr in ocr_results:
        for i in range(1, max_subjects+1):
            th_val = ocr.get(f"TH{i}")
            pr_val = ocr.get(f"PR{i}")

            # Clean TH and PR values: replace '*' and non-digit characters
            def parse_digit(val):
                if val is None:
                    return 0
                val = str(val).replace("*", "").strip()
                try:
                    return int(val)
                except ValueError:
                    return 0

            th_num = parse_digit(th_val)
            pr_num = parse_digit(pr_val)
            total = th_num + pr_num
            # Only update TOT if it's None or different
            if ocr.get(f"TOT{i}") is None or str(ocr.get(f"TOT{i}")) != str(total):
                ocr[f"TOT{i}"] = str(total).zfill(2) if total != 0 else None

    return ocr_results


def update_row_from_ocr(row, ocr, dob=False):
    # mapping between OCR keys and DB keys
    if not dob:
        key_map = {
            "SYMBOL" : "SYMBOL",
            "REG.NO.": "REG_NO",
            "NAME OF THE STUDENT": "NAME_OF_THE_STUDENTS",
            "REM": "RESULT",
            "DOB": "DOB",
            "TOTAL": "TOTAL",
        }
        row['QC_REMARKS'] = '[0]'
        row['QC_CHECK'] = 'Pass'
        # Then handle subjects (CODE, TH, PR, TOT)
        for i in range(1, 8):  # assuming up to 7 subjects
            for field in ["CODE", "TH", "PR", "TOT"]:
                ocr_key = f"{field}{i}"
                db_key = f"{field}{i}"
                if ocr_key in ocr:
                    row[db_key] = ocr[ocr_key]
    else:
        key_map = {
            "DOB": "DOB",
        }

    # First update direct mappings (like SYMBOL, DOB, TOTAL, etc.)
    for ocr_key, db_key in key_map.items():
        if ocr_key in ocr and ocr[ocr_key] not in (None, ""):
            if row[db_key] != ocr[ocr_key]:
                row[db_key] = ocr[ocr_key]
            elif dob:
                return None

    return row


def set_remarks(ocr_result_fun, out=False):
    """
    Compute QC_REMARKS for essential fields only:
    - SYMBOL
    - REG_NO
    - NAME OF THE STUDENT
    - DOB
    - TOTAL
    - TH, PR, TOT for subjects
    """
    def CombinationSum(subject_totals):
        partial_sums = set()
        # skip 1 subject
        if len(subject_totals) > 1:
            for combo in combinations(subject_totals, len(subject_totals) - 1):
                partial_sums.add(sum(combo))
        # skip 2 subjects (if at least 3 subjects exist)
        if len(subject_totals) > 2:
            for combo in combinations(subject_totals, len(subject_totals) - 2):
                partial_sums.add(sum(combo))
        return partial_sums
    
    reg_no = ocr_result_fun.get("REG.NO.") or ocr_result_fun.get("REG_NO")
    name   = ocr_result_fun.get("NAME OF THE STUDENT") or ocr_result_fun.get("NAME_OF_THE_STUDENTS")
    symbol = ocr_result_fun.get("SYMBOL")
    dob    = ocr_result_fun.get("DOB")
    total  = ocr_result_fun.get("TOTAL")
    result = ocr_result_fun.get("REM") or ocr_result_fun.get("RESULT")  # if exists
    school_name = ocr_result_fun.get("School_Name")
    school_code = ocr_result_fun.get("School_Code")
    ocr = bool(ocr_result_fun.get("REG.NO."))
    if ocr:
        school_code = "test"
        school_name = "test"
    QC_REMARKS = []
    subject_totals = []

    temp_total = 0
    for i in range(1, 8):
        code = ocr_result_fun.get(f"CODE{i}") or ""
        th = ocr_result_fun.get(f"TH{i}") or ""
        pr = ocr_result_fun.get(f"PR{i}") or ""
        tot = ocr_result_fun.get(f"TOT{i}") or ""

        # Remove '*' but keep letters like 'A', 'W'
        th_clean = str(th).replace("*", "").strip()
        pr_clean = str(pr).replace("*", "").strip()
        tot_clean = str(tot).replace("*", "").strip()

        # Convert to int only if numeric
        th_val = int(th_clean) if th_clean.isdigit() else 0
        pr_val = int(pr_clean) if pr_clean.isdigit() else 0
        tot_val = int(tot_clean) if tot_clean.isdigit() else 0
        temp_total += th_val + pr_val
        subject_totals.append(th_val + pr_val)

        if code and not th_clean and pr_val != tot_val:
            QC_REMARKS.append(7)

    # --- 0: Pass ---
    tot_ocr = int(ocr_result_fun.get("TOTAL") or 0)
    th_values = [ocr_result_fun.get(f"TH{i}") for i in range(1, 8)]
    code_values = [ocr_result_fun.get(f"CODE{i}") for i in range(1, 8)]
    if (
        reg_no not in (None, "") and
        name not in (None, "") and
        symbol not in (None, "") and
        dob not in (None, "") and
        total not in (None, "") and
        result not in (None, "") and
        school_name not in (None, "") and
        school_code not in (None, "") and
        (temp_total == tot_ocr or tot_ocr in CombinationSum(subject_totals))
    ):
        QC_REMARKS.append(0)
    else:
        if reg_no in (None, "") or ocr_result_fun.get("SYMBOL") in (None, ""):
            QC_REMARKS.append(1)
        if name in (None, "") or dob in (None, ""):
            QC_REMARKS.append(2)
        if all(v in (None, "") for v in th_values) or all(v in (None, "") for v in code_values):
            QC_REMARKS.append(3)
        if total in (None, ""):
            QC_REMARKS.append(4)
        if result in (None, ""):
            QC_REMARKS.append(5)
        if school_name in (None, "") or school_code in (None, ""):
            QC_REMARKS.append(6)
        if temp_total != tot_ocr or tot_ocr not in CombinationSum(subject_totals):
            QC_REMARKS.append(7)

    QC_REMARKS = list(set(QC_REMARKS))
    if out:
        return QC_REMARKS, "Pass" if QC_REMARKS == [0] else "Fail"
    else:
        return  (QC_REMARKS == [0])


def filter_rowdected(rows_detected, min_group_size = 3, gap_threshold = 30):
    if len(rows_detected) == 0:
        return []
    groups = []
    current_group = [rows_detected[0]]

    for r in rows_detected[1:]:
        if r - current_group[-1] <= gap_threshold:
            current_group.append(r)
        else:
            groups.append(current_group)
            current_group = [r]

    groups.append(current_group)  # add last group

    # Step 2: Filter groups with less than 3 members
    filtered_groups = [g for g in groups if len(g) >= min_group_size]

    # Step 3: Flatten to get a single array like rows_detected
    filtered_rows_detected = np.array([r for g in filtered_groups for r in g])

    return filtered_rows_detected

def deskew_image(img, debug=False):
    """
    Automatically rotate the image so that text is horizontal.

    Args:
        img: Input BGR image (numpy array).
        debug: If True, saves a debug image showing rotation.

    Returns:
        Rotated image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get text as foreground
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        # No text detected, return original
        if debug:
            print("No text detected, returning original image")
        return img.copy()

    # Compute minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if debug:
        print(f"Detected skew angle: {angle:.2f} degrees")

    # Rotate image to deskew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    if debug:
        cv2.imwrite("output_steps/deskew_debug.jpg", rotated)

    return rotated
