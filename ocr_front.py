from ocr_done_again.helper_using import NEB_OCR

def get_ocr_result(img):
    row = NEB_OCR(img)
    print(row)

get_ocr_result('D:/Neb_Ocr_Final/402.jpg')
