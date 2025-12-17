import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 전역 변수로 vectorstore와 embeddings 저장
_vectorstore = None
_embeddings = None

def get_embeddings():
    global _embeddings
    
    # 이미 로드된 임베딩이 있으면 재사용
    if _embeddings is not None:
        print("[Debug] 기존 임베딩 모델 재사용")
        return _embeddings
    
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    print(f"[Debug] 임베딩 모델 로드: {model_name}")
    
    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return _embeddings

def create_vector_db(full_text):
    global _vectorstore
    
    print("\n[VectorDB] 텍스트 청킹 시작...")

    embeddings = get_embeddings()

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    if len(full_text) < 50:
        print("[Warning] 텍스트가 너무 짧습니다.")
        docs = []
    else:
        texts = text_splitter.split_text(full_text)
        docs = [Document(page_content=t) for t in texts]
    
    print(f"  - 전체 텍스트 길이: {len(full_text)}자")
    print(f"  - Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"  - 생성된 청크 개수: {len(docs)}개")
    
    # 🔥 핵심: 명시적으로 메모리 기반 클라이언트 생성
    print("[VectorDB] EphemeralClient 생성 중...")
    client = chromadb.EphemeralClient()
    print("[VectorDB] EphemeralClient 생성 완료!")
    
    # 메모리 기반 ChromaDB 생성
    print("[VectorDB] Chroma vectorstore 생성 중...")
    _vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="investment_strategies",
        client=client  # 🔥 반드시 필요!
    )
    
    print(f"[VectorDB] ✅ 메모리 기반 저장 완료! (EphemeralClient)")
    return _vectorstore

def search_strategy(query, k=3):
    global _vectorstore
    
    print(f"\n[Search] 검색 질의: '{query}'")
    
    if _vectorstore is None:
        raise ValueError("[Error] VectorStore가 초기화되지 않았습니다. create_vector_db()를 먼저 호출하세요.")
    
    # 메모리에서 직접 검색
    results = _vectorstore.similarity_search(query, k=k)
    
    print(f"[Search] 검색 결과 {len(results)}건:")
    for i, res in enumerate(results):
        preview = res.page_content[:80].replace('\n', ' ')
        print(f"  [{i+1}] {preview}...")
    
    return results

def reset_db():
    global _vectorstore, _embeddings
    _vectorstore = None
    _embeddings = None
    print("[Info] 메모리 DB 및 임베딩 초기화 완료")