import cv2
import numpy as np

def columns(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return main_col_operation(binary)

def main_col_operation(binary,
                       cropped=None,
                       proj_thresh_ratio=0.05,
                       min_gap_width=15,
                       min_col_distance=8,
                       remove_vertical_lines=False,
                       vertical_kernel_height_factor=0):
    """
    Robust detection of column x-coordinates (safe_x) using vertical projection.

    Parameters
    ----------
    binary : np.ndarray
        Single-channel binary image (0/255 or 0/1). Foreground may be black (0) or white (255).
    proj_thresh_ratio : float
        Threshold ratio applied to the max column projection to decide 'empty' columns.
        Lower -> fewer columns considered empty.
    min_gap_width : int
        Minimum width (in pixels) for a gap to be accepted.
    min_col_distance : int
        Minimum distance between returned x positions; closer ones are merged.
    remove_vertical_lines : bool
        If True, attempt to detect and remove long vertical separators before projection.
    vertical_kernel_height_factor : int
        Heuristic factor to compute vertical kernel size for extracting vertical lines:
        kernel_height = max(3, height // vertical_kernel_height_factor).

    Returns
    -------
    lines_x : list[int]
        Sorted list of x coordinates (ints) marking safe column positions; includes 0 and width-1.
    """

    # normalize single-channel binary to boolean foreground mask
    if binary.ndim == 3:
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary.copy()

    # detect background value (which occurs more often)
    count0 = int((gray == 0).sum())
    count255 = int((gray == 255).sum())
    if count255 >= count0:
        # background likely white(255), foreground black(0)
        foreground = (gray == 0)
    else:
        # background likely black(0), foreground white(255)
        foreground = (gray > 0)

    height, width = gray.shape[:2]

    # Optionally remove long vertical separator lines (extract & subtract them)
    if remove_vertical_lines:
        # convert to uint8 for morphology
        fg_uint8 = (foreground.astype(np.uint8)) * 255
        k_h = max(3, height // vertical_kernel_height_factor)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        # opening extracts vertical long thin components
        vertical_lines = cv2.morphologyEx(fg_uint8, cv2.MORPH_OPEN, vertical_kernel)
        # subtract vertical_lines from foreground
        clean_fg_uint8 = cv2.subtract(fg_uint8, vertical_lines)
        foreground = (clean_fg_uint8 > 0)

    # vertical projection: count foreground pixels per column
    col_sum = np.sum(foreground, axis=0)  # length == width

    # threshold to decide which columns are 'empty' (gaps)
    max_col = float(col_sum.max()) if col_sum.size else 0.0
    threshold = max(1.0, max_col * proj_thresh_ratio)

    gap_mask = col_sum <= threshold  # True => column considered empty/available

    # find contiguous gap segments and accept those >= min_gap_width
    lines_x = []
    inside = False
    for x in range(width):
        if gap_mask[x] and not inside:
            start = x
            inside = True
        elif (not gap_mask[x]) and inside:
            end = x - 1
            inside = False
            if (end - start + 1) >= min_gap_width:
                safe_x = (start + end) // 2
                lines_x.append(int(safe_x))
    # trailing gap
    if inside:
        end = width - 1
        if (end - start + 1) >= min_gap_width:
            safe_x = (start + end) // 2
            lines_x.append(int(safe_x))

    # ensure boundaries exist
    if len(lines_x) == 0 or lines_x[0] > 0:
        lines_x = [0] + lines_x
    if len(lines_x) == 0 or lines_x[-1] < width - 1:
        lines_x = lines_x + [width - 1]

    # if cropped is not None:
    #     preview = cropped.copy()
    #     for x in lines_x:
    #         cv2.line(preview, (x, 0), (x, cropped.shape[0]), (0, 0, 255), 1)
    #     cv2.imwrite("output_steps/even_mergedlines.jpg", preview)
    # merge very-close x positions (caused by tiny splits)

    merged = []
    for x in sorted(lines_x):
        if not merged:
            merged.append(x)
        else:
            if x - merged[-1] <= min_col_distance:
                # merge by averaging
                merged[-1] = (merged[-1] + x) // 2
            else:
                merged.append(x)

    # final dedupe & sort
    final = sorted(set(int(x) for x in merged))
    return final


def columns_from_rows(cropped, filtered_row_lines, num_rows=7):
    """
    Detect vertical lines using only the first `num_rows` from filtered_row_lines.
    """
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # --- Determine vertical slice using first num_rows ---
    if len(filtered_row_lines) >= num_rows:
        top = filtered_row_lines[0]
        bottom = filtered_row_lines[num_rows - 1]
    else:
        top = filtered_row_lines[0]
        bottom = filtered_row_lines[-1]

    # Crop binary image to just these rows
    cropped_rows = binary[top:bottom, :]

    # --- Existing columns() logic on cropped_rows ---
    contours, _ = cv2.findContours(cropped_rows, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 8:
            boxes.append((x, y, w, h))

    height, width = cropped_rows.shape[:2]
    blocked = np.zeros(width, dtype=bool)
    buffer = 9
    for (x, y, w, h) in boxes:
        start = max(0, x - buffer)
        end = min(width, x + w + buffer)
        blocked[start:end+1] = True

    # --- Find free vertical gaps ---
    lines_x = []
    inside_gap = False
    for col in range(width):
        if not blocked[col] and not inside_gap:
            gap_start = col
            inside_gap = True
        elif blocked[col] and inside_gap:
            gap_end = col
            inside_gap = False
            safe_x = (gap_start + gap_end) // 2
            lines_x.append(safe_x)
    if inside_gap:
        safe_x = (gap_start + width) // 2
        lines_x.append(safe_x)

    return lines_x


def columns_for_different(cropped, after_line_index, before_line_index, filtered_row_lines, row_lines, debug=True):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    top_line = row_lines[0]
    row_lines_local = [y - top_line for y in row_lines]
    before_rows = []
    # if before_line_index >= 2:
    #     before_rows.append((filtered_row_lines[before_line_index-2], filtered_row_lines[before_line_index-1]))
    #     before_rows.append((filtered_row_lines[before_line_index-1], filtered_row_lines[before_line_index]))
    after_rows = []
    for i in range(after_line_index, len(filtered_row_lines)-1):
        after_rows.append((filtered_row_lines[i], filtered_row_lines[i+1]))
    rows_of_interest = before_rows + after_rows
    even_rows = [row for i, row in enumerate(rows_of_interest) if i % 2 == 0]
    odd_rows  = [row for i, row in enumerate(rows_of_interest) if i % 2 == 1]
    odd_mask  = binary.copy()
    even_mask = binary.copy()
    for (top, bottom) in even_rows:
        mask = odd_mask[top:bottom, :]
        mask[mask == 255] = 0
        odd_mask[top:bottom, :] = mask
    for (top, bottom) in odd_rows:
        mask = even_mask[top:bottom, :]
        mask[mask == 255] = 0
        even_mask[top:bottom, :] = mask
     # --- New code: make row_lines_local black with height 5 ---
    height = 12
    for y in row_lines_local:
        y_start = max(0, y - height // 2)
        y_end = min(binary.shape[0], y + height // 2 + 1)
        odd_mask[y_start:y_end, :] = 0
        even_mask[y_start:y_end, :] = 0
    # ---------------------------------------------------------

    # Step 6: save final cropped images
    if debug:
        cv2.imwrite("output_steps/odd.jpg", odd_mask)
        cv2.imwrite("output_steps/even.jpg", even_mask)

    return odd_mask, even_mask


def merged_col_operation(cropped, after_line_index, before_line_index, filtered_row_lines, row_lines, debug=True):
    # Step 1: Create odd_mask and even_mask
    odd_mask, even_mask = columns_for_different(cropped, after_line_index, before_line_index, filtered_row_lines, row_lines)

    # Step 2: Run main_col_operation on both masks
    odd_lines  = main_col_operation(odd_mask)
    even_lines = main_col_operation(even_mask, cropped)
    if debug:
        preview = cropped.copy()
        for x in odd_lines:
            cv2.line(preview, (x, 0), (x, cropped.shape[0]), (0, 0, 255), 1)
        cv2.imwrite("output_steps/odd_lines.jpg", preview)
        preview = cropped.copy()
        for x in even_lines:
            cv2.line(preview, (x, 0), (x, cropped.shape[0]), (0, 0, 255), 1)
        cv2.imwrite("output_steps/even_lines.jpg", preview)

    # Step 3: Build final lines per row
    row_indices  = []
    rows_of_interest = []
    # if before_line_index >= 2:
    #     rows_of_interest.append((filtered_row_lines[before_line_index-2], filtered_row_lines[before_line_index-1]))
    #     row_indices.append(before_line_index-2)
    #     rows_of_interest.append((filtered_row_lines[before_line_index-1], filtered_row_lines[before_line_index]))
    #     row_indices.append(before_line_index-1)
    for i in range(after_line_index, len(filtered_row_lines)-1):
        rows_of_interest.append((filtered_row_lines[i], filtered_row_lines[i+1]))
        row_indices.append(i)
    if debug:
        vis = cropped.copy()
    row_vlines = {}
    for idx, (top, bottom) in enumerate(rows_of_interest):
        row_idx = row_indices[idx]
        lines_x = even_lines if idx % 2 == 0 else odd_lines
        row_vlines[row_idx] = lines_x  # store per row
        if debug:
            for x in lines_x:
                cv2.line(vis, (x, top), (x, bottom), (0, 0, 255), 1)  # red lines

    # Step 6: Save / return result
    if debug:
        cv2.imwrite("output_steps/merged_lines_on_cropped.jpg", vis)

    return row_vlines
