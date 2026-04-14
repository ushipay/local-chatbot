from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


def build_index() -> None:
    """docsディレクトリのMarkdownからFAISSインデックスを構築して保存する。"""
    loader = DirectoryLoader(
        str(DOCS_DIR), glob="*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Index built: {len(chunks)} chunks saved to {INDEX_DIR}")


def load_index() -> FAISS:
    """保存済みFAISSインデックスを読み込んで返す。"""
    return FAISS.load_local(
        str(INDEX_DIR), _get_embeddings(), allow_dangerous_deserialization=True,
    )
