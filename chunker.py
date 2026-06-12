import re

# --- Mathematical Guardrail Setup ---
MAX_TOKENS = 384
OVERLAP_TOKENS = 50

def approximate_tokens(text):
    return int(len(text.split()) * 1.3)

def chunk_with_overlap(text, max_t=MAX_TOKENS, overlap_t=OVERLAP_TOKENS):
    words = text.split()
    chunks = []
    
    # counting the number of new words (overlapping) words are shared with previous chunk to preserve context to know the step we move with inside the loop
    # count also the chunk size in terms of words roughly tokens / 1.3 as word contain approx 1.3 token
    step = max(1, int((max_t - overlap_t) / 1.3))
    chunk_size = int(max_t / 1.3)
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
    return chunks


def semantic_chunk_cv(raw_text):
    headers_pattern = r"^(Contact|Top Skills|Summary|Experience|Education|Certifications|Languages|Publications|Projects|Honors & Awards|Honors-Awards)\s*$"
    
    parts = re.split(headers_pattern, raw_text, flags=re.MULTILINE)
    final_chunks = []
    
    # checking for into unnamed section of cv
    candidate_name = "Unknown"
    if parts[0].strip():
        candidate_name = parts[0].strip().split('\n')[0].strip()
        print(f"------ {candidate_name}")
        final_chunks.append({
            "metadata": {
                "section": "Intro/Profile"
                , "candidate_name": candidate_name
            }, 
            "text": parts[0].strip()
        })
  
        

    # loop over remaining parts (Header -> Content -> Header -> Content) alteration
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        section_content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        if not section_content:
            continue
            
        if approximate_tokens(section_content) > MAX_TOKENS:
            # breaking the long section into sub chunks to be valid size for embedding model
            sub_chunks = chunk_with_overlap(section_content)
            for idx, sub_text in enumerate(sub_chunks):
                final_chunks.append({
                    "metadata": {"section": f"{section_name} (Part {idx+1})"
                                 , "candidate_name": candidate_name
                                 },
                    "text": sub_text
                })
        else:
            final_chunks.append({
                "metadata": {"section": section_name
                             ,"candidate_name": candidate_name},
                "text": section_content
            })
            
    return final_chunks

