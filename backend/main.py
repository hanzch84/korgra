# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
import heapq
from datetime import datetime

# 루트 폴더의 .env 파일 로드
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="Korean Word Graph API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
openai_client = None
word_vectors = {}
word_cache = {}

# 우선순위 큐 (힙큐)
word_queue = []  # Min-heap: (timestamp, student_id, word, submission)
processed_words = set()  # 중복 처리 방지

# 한국어 임베딩 모델 로드
def load_korean_model():
    global model
    try:
        # KoSentenceBERT 모델 로드 (한국어 특화)
        model_name = "jhgan/ko-sroberta-multitask"
        model = SentenceTransformer(model_name)
        logger.info(f"Korean embedding model loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load Korean model: {e}")
        # 백업 모델
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Fallback to multilingual model")

# OpenAI 클라이언트 초기화
def initialize_openai():
    global openai_client
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OpenAI API key not found")

# 데이터 모델
class WordInput(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class ClusterRequest(BaseModel):
    words: List[str]
    num_clusters: Optional[int] = None

class LabelRequest(BaseModel):
    words: List[str]

class HybridSimilarityResponse(BaseModel):
    vector_similarity: float
    llm_similarity: float
    hybrid_score: float
    confidence: str

class ClusterResponse(BaseModel):
    cluster_assignments: List[int]
    cluster_centers: List[List[float]]
    silhouette_score: float

# 앱 시작시 모델 로드
@app.on_event("startup")
async def startup_event():
    load_korean_model()
    initialize_openai()

# 학생 단어 제출 접수
@app.post("/api/submit-word")
async def submit_word_to_queue(request: dict):
    #학생이 제출한 단어를 우선순위 큐에 추가
    try:
        word = request.get("word", "").strip()
        student_id = request.get("studentId", "anonymous")
        timestamp = request.get("timestamp", int(datetime.now().timestamp() * 1000))
        
        if not word:
            raise HTTPException(status_code=400, detail="단어가 필요합니다")
        
        # 우선순위 큐에 추가 (timestamp가 작을수록 우선순위 높음)
        submission = {
            "word": word,
            "student_id": student_id,
            "timestamp": timestamp,
            "received_at": datetime.now().isoformat()
        }
        
        # 힙큐에 추가 (timestamp 기준 정렬)
        heapq.heappush(word_queue, (timestamp, student_id, word, submission))
        
        logger.info(f"Word queued: {word} from {student_id} at {timestamp}")
        
        return {
            "status": "queued",
            "word": word,
            "student_id": student_id,
            "queue_position": len(word_queue),
            "timestamp": timestamp
        }
    
    except Exception as e:
        logger.error(f"Submit word error: {e}")
        raise HTTPException(status_code=500, detail=f"단어 제출 실패: {str(e)}")

# 큐에서 다음 단어 가져오기
@app.get("/api/next-word")
async def get_next_word():
    """우선순위 큐에서 다음 처리할 단어 반환"""
    try:
        if not word_queue:
            return {
                "status": "empty",
                "queue_size": 0,
                "message": "처리할 단어가 없습니다"
            }
        
        # 가장 오래된 단어 추출
        timestamp, student_id, word, submission = heapq.heappop(word_queue)
        
        # 중복 처리 방지
        word_key = f"{word}-{student_id}"
        if word_key in processed_words:
            # 중복이면 다음 단어로
            if word_queue:
                return await get_next_word()
            else:
                return {"status": "empty", "queue_size": 0}
        
        processed_words.add(word_key)
        
        return {
            "status": "word_ready",
            "word": word,
            "student_id": student_id,
            "timestamp": timestamp,
            "submission": submission,
            "queue_size": len(word_queue)
        }
    
    except Exception as e:
        logger.error(f"Get next word error: {e}")
        raise HTTPException(status_code=500, detail=f"다음 단어 가져오기 실패: {str(e)}")

# 큐 상태 확인
@app.get("/api/queue-status")
async def get_queue_status():
    """현재 큐 상태 반환"""
    try:
        # 큐에서 미리보기 (실제로 제거하지 않음)
        preview = []
        temp_queue = word_queue.copy()
        
        for i in range(min(5, len(temp_queue))):
            if temp_queue:
                timestamp, student_id, word, submission = temp_queue[i]
                preview.append({
                    "word": word,
                    "student_id": student_id,
                    "timestamp": timestamp,
                    "wait_time": int((datetime.now().timestamp() * 1000 - timestamp) / 1000)
                })
        
        return {
            "queue_size": len(word_queue),
            "preview": preview,
            "processed_count": len(processed_words)
        }
    
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        raise HTTPException(status_code=500, detail=f"큐 상태 확인 실패: {str(e)}")

# 빠른 LLM 유사도 계산 (최적화)
@app.post("/api/quick-similarity")
async def quick_llm_similarity(request: dict):
    """빠른 LLM 유사도 계산 - 간결한 프롬프트로 속도 최적화"""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI API 사용 불가")
        
        word1 = request.get("word1", "")
        word2 = request.get("word2", "")
        
        # 매우 간결한 프롬프트로 빠른 응답
        prompt = f'"{word1}"와 "{word2}" 연관성 점수 (0-100): '
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,  # 토큰 수 최소화
            temperature=0.1  # 일관성 있는 답변
        )
        
        return {
            "word1": word1,
            "word2": word2,
            "response": response.choices[0].message.content.strip(),
            "method": "quick_llm"
        }
    
    except Exception as e:
        logger.error(f"Quick LLM similarity error: {e}")
        raise HTTPException(status_code=500, detail=f"빠른 LLM 유사도 실패: {str(e)}")

# 한국어 단어 벡터화
@app.post("/api/vectorize")
async def vectorize_word(word_input: WordInput):
    """한국어 단어를 벡터로 변환"""
    try:
        word = word_input.word.strip()
        
        # 캐시 확인
        if word in word_vectors:
            return {
                "word": word,
                "vector": word_vectors[word].tolist(),
                "cached": True
            }
        
        # 벡터화
        vector = model.encode([word])[0]
        word_vectors[word] = vector
        
        return {
            "word": word,
            "vector": vector.tolist(),
            "cached": False,
            "vector_dim": len(vector)
        }
    
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail=f"벡터화 실패: {str(e)}")

# 벡터 유사도 계산
@app.post("/api/vector-similarity")
async def calculate_vector_similarity(request: SimilarityRequest):
    """두 단어의 벡터 유사도 계산"""
    try:
        # 두 단어 벡터화
        word1_vector = model.encode([request.word1])[0]
        word2_vector = model.encode([request.word2])[0]
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([word1_vector], [word2_vector])[0][0]
        
        return {
            "word1": request.word1,
            "word2": request.word2,
            "similarity": float(similarity),
            "method": "cosine_similarity"
        }
    
    except Exception as e:
        logger.error(f"Vector similarity error: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 유사도 계산 실패: {str(e)}")

# K-means 클러스터링
@app.post("/api/clustering", response_model=ClusterResponse)
async def perform_clustering(request: ClusterRequest):
    """한국어 단어들의 K-means 클러스터링"""
    try:
        if len(request.words) < 2:
            raise HTTPException(status_code=400, detail="최소 2개 이상의 단어가 필요합니다")
        
        # 단어들 벡터화
        vectors = model.encode(request.words)
        
        # 클러스터 수 결정
        if request.num_clusters is None:
            num_clusters = min(max(2, len(request.words) // 3), 8)
        else:
            num_clusters = min(request.num_clusters, len(request.words))
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(vectors).tolist()
        
        # 실루엣 점수 계산
        from sklearn.metrics import silhouette_score
        if len(set(cluster_assignments)) > 1:
            sil_score = float(silhouette_score(vectors, cluster_assignments))
        else:
            sil_score = 0.0
        
        return ClusterResponse(
            cluster_assignments=cluster_assignments,
            cluster_centers=kmeans.cluster_centers_.tolist(),
            silhouette_score=sil_score
        )
    
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        raise HTTPException(status_code=500, detail=f"클러스터링 실패: {str(e)}")

# LLM 기반 클러스터 라벨링
@app.post("/api/generate-labels")
async def generate_cluster_labels(request: LabelRequest):
    """LLM을 활용한 클러스터 라벨 생성"""
    try:
        if not openai_client:
            # OpenAI 없을 때 기본 라벨링
            return {"label": f"그룹 ({len(request.words)}개 단어)", "method": "fallback"}
        
        words_str = ", ".join(request.words)
        
        prompt = f"""
다음 한국어 단어들의 공통 주제나 카테고리를 한 단어 또는 짧은 구문으로 요약해주세요.

단어들: {words_str}

가능한 카테고리 예시:
- 동물, 스포츠, 음식, 가족, 감정, 색깔, 자연, 교통, 전자기기, 학교 등

적절한 이모지와 함께 답해주세요 (예: 🐾 동물, ⚽ 스포츠):
"""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        
        label = response.choices[0].message.content.strip()
        
        return {
            "words": request.words,
            "label": label,
            "method": "gpt-3.5-turbo"
        }
    
    except Exception as e:
        logger.error(f"Label generation error: {e}")
        # 실패시 기본 라벨
        return {
            "words": request.words,
            "label": f"📊 그룹 {len(request.words)}",
            "method": "fallback",
            "error": str(e)
        }

# 헬스체크
@app.get("/api/health")
async def health_check():
    """API 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "openai_available": openai_client is not None,
        "cached_words": len(word_vectors),
        "cached_similarities": len(word_cache),
        "queue_size": len(word_queue)
    }

# 메인 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)