import os
from cv_parser import extractTextFromPDF
from vectorizer import build_vector_db

def process_all_cvs(directory_path):
    raw_cv_texts = []
    
    print(f"scanning '{directory_path}' for pdf files")
    

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"extracting text from: {filename}")
            
            extracted_text = extractTextFromPDF(file_path)
            
            if extracted_text:
                raw_cv_texts.append(extracted_text)
            else:
                print(f"warning: Could not extract text from {filename}")
                
    print(f"\n sucess extract text from {len(raw_cv_texts)} CVs.")
    print("starting vectorization \n")
    
    if raw_cv_texts:
        build_vector_db(raw_cv_texts)
    else:
        print("no text extracted - check your PDF folder.")

if __name__ == "__main__":
    cv_directory = "./Dataset" 
    
    if not os.path.exists(cv_directory):
        print(f"error: The directory '{cv_directory}' does not exist.")
    else:
        process_all_cvs(cv_directory)