import os
from docx import Document
from PyPDF2 import PdfReader
from typing import List
def load_documents(folder_path: str) ->List[str]:
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            with open(file_path,"r",encoding="utf-8") as f:
                docs.append(f.read())
        elif filename.endswith(".pdf"):
            text =""
            reader = PdfReader(file_path)
            for page in reader.pages:
                text+=page.extract_text() + "\n"
            docs.append(text)

        elif filename.endswith(".docs"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraph])
            docs.append(text)
    
    return docs