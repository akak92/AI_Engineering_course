import re
import tiktoken
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        # Configuramos el splitter recursivo: intenta dividir por parrafos, 
        # luego oraciones, luego palabras
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Tamaño máximo de cada chunk
            chunk_overlap=50,  # Superposición entre chunks
            length_function=self.calculate_tokens,  # Función para contar tokens
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto base elimninando ruido innecesario
        """
        # Saltos de linea múltiple y espacios extra
        text = re.sub(r'\s+', ' ', text)
        # Eliminamos caracteres especiales repetitivos (comunes en PDFs mal extraídos)
        text = re.sub(r'(\.\s){2,}', '. ', text)
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """
        Calcula la cantidad de tokens exactos usando tiktoken
        """
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline principal: Limpieza -> Chunking -> Validación."""
        # 1. Limpieza
        cleaned_text = self.clean_text(raw_text)
        # 2. Chunking (fragmentación)
        chunks = self.splitter.split_text(cleaned_text)
        # 3. Validación de chunks
        for i, chunk in enumerate(chunks):
            token_count = self.calculate_tokens(chunk)
            print(f"Chunk {i+1}: {token_count} tokens")
        return chunks

