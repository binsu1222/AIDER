import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DB가 저장될 로컬 경로
PERSIST_DIRECTORY = "./chroma_db"

def get_embeddings():
    # 임베딩 모델 설정을 한 곳에서 관리
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def create_vector_db(full_text):
    # 텍스트를 청킹하여 ChromaDB에 저장
    print("\n[Chunking] Recursive Character Text Splitter !")

    embeddings = get_embeddings()

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""], # 분할 기준으로 사용할 문자 목록
        length_function=len
    )

    if len(full_text) < 50:
        print("[Warning] 텍스트가 너무 짧습니다.")
        docs = []
    else:
        texts = text_splitter.split_text(full_text)
        from langchain_core.documents import Document
        docs = [Document(page_content=t) for t in texts]
    
    print(f"  - 전체 텍스트 길이: {len(full_text)}자")
    print(f"  - (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    print(f"  - 생성된 청크(Chunk) 개수: {len(docs)}개")
    
    # ChromaDB 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="investment_strategies" # 컬렉션 이름 지정
    )
    
    print(f"[ChromaDB 저장] 데이터가 '{PERSIST_DIRECTORY}' 폴더에 저장 !")
    return vectorstore

def search_strategy(query, k=3):
    # 저장된 ChromaDB에서 질문과 유사한 내용을 검색
    print(f"\n[Search] 검색 질의: '{query}'")
    
    # 1. 임베딩 모델 다시 로드 (저장할 때와 동일해야 함)
    embeddings = get_embeddings()
    
    # 2. 디스크에 저장된 DB 불러오기
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="investment_strategies"
    )
    
    # 3. 유사도 검색
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"[Search] 검색 결과({len(results)}건):")
    for i, res in enumerate(results):
        print(f"  {i+1}. {res.page_content[:100]}... (생략)")
        print("-" * 30)
    
    return results

def reset_db():
    # DB 초기화시 사용
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("[Info] 기존 DB 삭제 완료")