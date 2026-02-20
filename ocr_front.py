from ocr_done_again.helper_using import NEB_OCR

def get_ocr_result(img):
    row = NEB_OCR(img)
    # row = NEB_OCR('D:/xampp/htdocs/neb/img/2070/2070 XII Reg Science/Book 3 2070 XII Reg Science 1716-2518/402.jpg')
    return row

# get_ocr_result('D:/xampp/htdocs/neb/img/2070/2070 XII Reg Science/Book 3 2070 XII Reg Science 1716-2518/402.jpg')
# get_ocr_result('D:/Neb_Ocr_Final/scanned_output.jpg')
