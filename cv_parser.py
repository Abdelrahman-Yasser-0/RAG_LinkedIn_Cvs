import fitz 
import os
import re

def extractTextFromPDF(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        
        main_body_text = ""
        sidebar_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            
            blocks = [b for b in blocks if not re.search(r"Page \d+ of \d+", b[4], re.IGNORECASE)]
            
            if page_num == 0:
                left_blocks = [b for b in blocks if b[0] < 220]
                right_blocks = [b for b in blocks if b[0] >= 220]
                
                left_blocks.sort(key=lambda b: b[1])
                right_blocks.sort(key=lambda b: b[1])
                
                sidebar_text += "\n".join([b[4] for b in left_blocks]) + "\n"
                main_body_text += "\n".join([b[4] for b in right_blocks]) + "\n"
            else:
                blocks.sort(key=lambda b: b[1])
                main_body_text += "\n".join([b[4] for b in blocks]) + "\n"
                
        
        full_text = main_body_text + "\n" + sidebar_text
   
        clean_text = re.sub(r'\n+', '\n', full_text).strip()
        return clean_text
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None