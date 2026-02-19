import cv2
import numpy as np
from core import extract_data
from detect_rows import rows, rows_full
from detect_columns import columns, columns_from_rows, columns_for_different, merged_col_operation
from utils import *
import matplotlib.pyplot as plt
import glob, os
import itertools

# output_folder = "output_steps"

# # Delete all files in the folder
# files = glob.glob(os.path.join(output_folder, "*"))
# for f in files:
#     try:
#         os.remove(f)
#     except Exception as e:
#         print(f"Could not remove {f}: {e}")

# Load image
# img = cv2.imread("D:/xampp/htdocs/neb/img/2067/2067 XII Reg Science/Book 4 2067 XII Reg Science 2730-2795/120.jpg")
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XI Supplementary\Book 3 2067 XI Supplemantary 0551-0820\180.jpg")
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XII Reg\Book 9 2067 XII Reg 0610-0623\246.jpg")
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XII Partial\Book 3 2067 XII Partial 0426-0509\60.jpg") # Dob mistake
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XII Partial\Book 12 2067 XII Partial 2101-2401\114.jpg") # Dob mistake/////
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XII Reg\Book 2 2067 XII Reg 0301-0322\340.jpg") # REG Problem
# img = cv2.imread(r"D:/xampp/htdocs/neb/img\2067\2067 XII Reg\Book 14 2067 XII Reg 1404-1505\1297.jpg")
# D:\xampp\htdocs\neb\img\2067\2067 XII Science Partial\Book 3 2067 XII Science Partial 5405-7712
# img = cv2.imread(r"D:\xampp\htdocs\neb\img\2067\2067 XII Science Partial\Book 3 2067 XII Science Partial 5405-7712\16.jpg")
img = cv2.imread(r"D:\xampp\htdocs\neb\img\2067\2067 XII Reg Science\Book 3 2067 XII  Reg Science 2601-2729\339.jpg")

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
cv2.imwrite("output_steps/step2_no_green.jpg", img_no_green)

# Removing if hight is > 35
gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
_, oldbinary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(oldbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_no_green = img_no_green.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h >= 35:  # Removing if hifht is > 35
        cv2.drawContours(img_no_green, [cnt], -1,  (255, 255, 255), -1)
cv2.imwrite("output_steps/step2_5_no_green_updated.jpg", img_no_green)

# --- STEP 2: Detect first and last row with sensitivity 0.18 ---
gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
_, oldbinary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(oldbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
binary = np.zeros_like(oldbinary)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h < 35:  # Removing if hifht is > 35
        cv2.drawContours(binary, [cnt], -1, 255, -1)
cv2.imwrite("output_steps/step3_binary.jpg", binary)
height, width = binary.shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 1))  # 5 pixels wide, 1 pixel high
binary_expanded = cv2.dilate(binary, kernel, iterations=1)
# --- Step 1: Compute row-wise sum (count white pixels per row) ---
row_sum_filtered = np.array([count_long_white_clusters(binary_expanded[y], min_gap=200)
                             for y in range(height)])
# --- Step 2: Sum over 5 consecutive rows for vertical smoothing ---
window_size = 10
row_sum_multi = np.convolve(row_sum_filtered, np.ones(window_size, dtype=int), mode='same')
threshold = np.max(row_sum_multi) * 0.18
rows_detected = np.where(row_sum_multi > threshold)[0]
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
        
        # Confirm the required distribution
        if top_half_count >= 2 and bottom_half_count >= 1:
            pass
            break  # good distribution, stop
    continuous_thresh -= 0.01   # step down threshold
printtest = img_no_green.copy()
for y in row_lines:
    cv2.line(printtest, (0, y), (img.shape[1], y), (0, 0, 255), 1)
cv2.imwrite("output_steps/step4_no_green.jpg", printtest)
# Crop between first and last detected line
top_line = row_lines[0]
bottom_line = row_lines[-1]
cropped = img_no_green[top_line:bottom_line, :]
cv2.imwrite("output_steps/step5_cropped.jpg", cropped)





# --- STEP 3: Detect rows and columns ---
gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
_, binary_cropped = cv2.threshold(gray_cropped, 150, 255, cv2.THRESH_BINARY_INV)
filtered_row_lines = rows_full(cropped, row_lines)
lines_x = columns(cropped)
h_lines = filtered_row_lines   # horizontal lines after filtering + shift
v_lines = lines_x              # vertical safe lines
binary_used = binary_cropped   # binary image of cropped region
cv2.imwrite("output_steps/step6_binary_cropped.jpg", binary_used)



# --- STEP 4: Detect first column ---
first_row_top = h_lines[0]
first_row_bottom = h_lines[1] if len(h_lines) > 1 else binary_used.shape[0]
first_row_region = binary_used[first_row_top:first_row_bottom, :]
col_densities = []
for i in range(len(v_lines)-1):
    left = v_lines[i]
    right = v_lines[i+1]
    box_region = first_row_region[:, left:right]
    # plt.imshow(box_region, cmap="gray")   # use gray colormap for binary images
    # plt.title("ROI")
    # plt.axis("off")
    # plt.show()
    density = np.sum(box_region > 0) / box_region.size
    col_densities.append(density)
# --- Remove last column if its density is 0 ---
if col_densities[-1] == 0:
    print("Removing last column with 0 density")
    v_lines = v_lines[:-1]
    col_densities = col_densities[:-1]
# --- Step 2: Find fixed_col_index with density > 0.05 ---
fixed_col_index = None
for i, density in enumerate(col_densities):
    if density > 0.045:
        fixed_col_index = i
        break
if fixed_col_index is None:
    fixed_col_index = 0  # fallback
print("Fixed column index:", fixed_col_index)


# --- STEP 4: Detect rows after between symbol and name image ---
# h_lines = rows(cropped, v_lines, fixed_col_index, row_lines)
# v_lines = columns_from_rows(cropped, h_lines)

# Draw horizontal lines on cropped image
preview = cropped.copy()
for y in h_lines:
    cv2.line(preview, (0, y), (preview.shape[1], y), (0, 0, 255), 1)
for x in v_lines:
    cv2.line(preview, (x, 0), (x, height), (0, 0, 255), 1)
cv2.imwrite("output_steps/step7_cropped_with_lines.jpg", preview)


# --- Step 5: Select boxes in rest of rows starting from fixed column ---
case=False
boxes_selected = get_selected_box(case, row_lines, h_lines, filtered_row_lines, cropped, fixed_col_index, v_lines)

print(len(boxes_selected))

# boxes_filtered = enforce_row_box_count(boxes_selected, y_tolerance=5, total_columns=24)
# print("Total boxes after enforcing per row:", len(boxes_filtered))
boxes_limited = filter_boxes_by_first_row_limits(boxes_selected, cropped)
print("Boxes after applying first row limits:", len(boxes_limited))

# Finding 

img_draw = cropped.copy()
for (x1, y1, x2, y2) in boxes_limited:
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 1)

cv2.imwrite("output_steps/step8_boxes_selected.jpg", img_draw)


# extract_data(cropped, boxes_limited)

boxes_in_original = []
for (x1, y1, x2, y2) in boxes_limited:
    boxes_in_original.append((x1, y1 + top_line, x2, y2 + top_line))

result_text = extract_data(img, boxes_in_original)
# result_text = ['18', 'SYMBOL', 'REG.NO.', 'NAME OF THE STUDENT', 'DOB', 'CODE', 'TH', 'CODE', 'TH', 'PR', 'TOT', 'CODE', 'TH', 'PR', 'TOT', 'CODE', 'TH', 'PR', 'TOT', 'CODE', 'TH', 'PR', 'TOT', 'CODE', 'TH', 'PR', 'TOT', 'CODE', 'TH', 'PR', 'TOT', 'TOTAL', 'REM', '25401222', '665401242', 'GITA NEPALI', '2049/', '/12/', '/06', 'C.ENG', '42', 'MATHS', '42', 'CHILD', '42', 'TCHSC', '29', 'TCHSS', '24', 'INSTEV', '20', '-PRACTI', '37', '236', 'PASS', '25401223', '665401243', 'GYANU DEVI ROKAYA', '2045', '/107', '701', 'C.ENG', '40', 'HL TH&', '09*', '43', '52', 'CHILO', '41', 'TCHSC', '21', 'TCHSS', '22', 'INSTEV', '23', 'PRACTI', '37', '236', 'FAIL1', '25401224', '665401244', 'HARI BAHADUR BUOHATHOKI', '2048/', '/05/', '/10', 'C.ENG', '35', 'ECO', '39', 'OMSP', '34', '23', '57', 'ACC', '45', 'BUSS', '54', '230', 'PASS', 'ITLA', '***', '25401225', '665401245', 'HARI PRASAD SHARMA', '2049', '/12/', '/02', 'C.ENG', '41', 'NEP', '50', 'CHILD', '45', 'TCHSC', '30', 'TCHSS', '30', 'INSTEV', '23', 'PRACTI', '35', '254', 'PASS', '25401226', '565401246', 'HARI BAHADUR K.C.', '2045/', '/03/', '/25', 'C.ENG', '41', 'ENG', '35', 'CHILD', '35', 'TCHSC', '27', 'TCHSS', '27', 'INSTEV', '20', 'PRACTI', '35', '220', 'PASS', '25401227', '665401247', 'HARI BAHADUR RANA', '2039', '/04)', '/10', 'C.ENG', '35', 'HL TH&', '12*', '37', '49', 'CHILD', '26*', 'TCHSC', '18', 'TCHSS', '12*', 'INSTEV', '11*', 'PRACTI', '37', '188', 'FAIL4', '25401228', '565401249', 'HEMRAJ MALLA', '20477', '710', '712', 'C.ENG', '47', 'ENG', '46', 'CHILO', '46', 'TCHSC', '27', 'TCHSS', '26', 'INSTEV', '21', 'PRACTI', '35', '248', 'PASS', '1140', 'AM', '25401229', '665401250', 'BHUM BAHADUR OLI', '2045/', '/05/', '701', 'C.ENG', '38', 'HL TH&', '13*', '43', '56', 'CHILD', '36', 'TCHSC', '21', 'TCHSS', '25', ':', 'INSTEV', '21', 'PRACTI', '37', '234', 'FAILI', '4', '25401230', '565401251', 'CHETANA BATALA', '20477', '/03/', '712', '***', 'A RUNPLE', '4MM', '1.50', '72.44', 'C.ENG', '35', 'HIST', '38', 'HL TH&', '18', '41', '59', 'INSTEV', 'FA', 'CHILD', '44', 'PRACTI', 'A-', '476', 'FAIL2', '25401231', '665401252', 'DEVRAJ SHARMA', '2045', '/05/', '16', 'C.ENG', '41', 'NEP', '47', 'CHILD', '44', 'TCHSC', '33', 'TCHSS', '19', 'INSTEV', '22', 'PRACTI', '38', '244', 'PASS', '25401232', '665401253', 'KAMALA BHANDARI', '2049', '7127', '102', '750', 'C.ENG', 'A', 'MATHS', 'A', 'CHILD', 'A', 'TCHSC', 'A', 'TCHSS', 'A', 'INSTEV', 'A', 'PRACTI', '36', '036', 'FAIL6', '1162', '***', '25401233', '665401254', 'KAMALA KUMARI B.K.', '2043/', '/03/', '710', 'C.ENG', '38', 'HL TH&', '10*', '43', '53', 'CHILD', '36', 'TCHSC', '22', 'TCHSS', '19.', 'INSTEV', '21', 'PRACTI', '37', '226', 'FAILI', '25401234', '665401255', 'NIRMAL BUDHATHOKI', '2047', '/05/', '/04', 'C.ENG', '39', 'ECO', '35', 'CHILD', '39', 'ENG', '35', 'RURAL', '35', '183', 'PASS']

boxes_filtered, texts_filtered = filter_columns_from_symbol_rem(boxes_limited, result_text)
rows_text = group_boxes_into_rows(boxes_filtered, texts_filtered)
normalized_rows  = normalize_text(rows_text)
data = build_flat_student_records(normalized_rows)

#  Removing the subject took from the writen rows (Mistaken)
cleaned_data = []
for stu in data:
    count = 0
    for k, v in stu.items():
        if k.startswith(('CODE')):
            count += 1
    if count <= 7:
        cleaned_data.append(stu)

for i, stu in enumerate(cleaned_data, 1):
    row = " | ".join(f"{k}={v}" for k, v in stu.items())
    print(f"Student {i}: {row}\n")

# for i, stu in enumerate(data, 1):
#     row = " | ".join(f"{k}={v}" for k, v in stu.items())
#     print(f"Student {i}: {row}")
#     print("")