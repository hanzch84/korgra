"""
한국어 NLP 의미적 연관 네트워크 - FastAPI 백엔드
실제 KoSentenceBERT, Word2Vec, ConceptNet 활용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
import httpx
import logging
from datetime import datetime

from nlp_models import (
    KoreanNLPManager,
    SemanticSimilarityCalculator,
    ClusteringEngine,
    GPTLabeler
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Korean NLP Semantic Network API",
    description="실시간 한국어 의미적 연관 네트워크 분석 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연결)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 - 실제 배포시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 NLP 매니저 인스턴스
nlp_manager = None
similarity_calculator = None
clustering_engine = None
gpt_labeler = None

# 요청/응답 모델 정의
class WordRequest(BaseModel):
    word: str

class WordPairRequest(BaseModel):
    word1: str
    word2: str

class ClusteringRequest(BaseModel):
    words: List[str]
    num_clusters: int = 3

class LabelingRequest(BaseModel):
    words: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    sbert_loaded: bool
    word2vec_loaded: bool
    conceptnet_available: bool
    gpt_available: bool

class VectorResponse(BaseModel):
    word: str
    vector: List[float]
    method: str

class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float
    method: str
    confidence: str

class HybridSimilarityResponse(BaseModel):
    word1: str
    word2: str
    hybrid_score: float
    vector_similarity: float
    llm_similarity: float
    conceptnet_score: float
    confidence: str

class ClusteringResponse(BaseModel):
    words: List[str]
    cluster_assignments: List[int]
    silhouette_score: float
    method: str

class LabelingResponse(BaseModel):
    words: List[str]
    label: str
    confidence: float

# 시작 이벤트 - NLP 모델 로딩
@app.on_event("startup")
async def startup_event():
    global nlp_manager, similarity_calculator, clustering_engine, gpt_labeler
    
    logger.info("🚀 한국어 NLP 백엔드 시작 중...")
    
    try:
        # NLP 매니저 초기화 (모델 로딩)
        logger.info("📦 KoSentenceBERT 및 Word2Vec 모델 로딩 중...")
        nlp_manager = KoreanNLPManager()
        await nlp_manager.initialize_models()
        
        # 유사도 계산기 초기화
        logger.info("🔍 의미적 유사도 계산기 초기화 중...")
        similarity_calculator = SemanticSimilarityCalculator(nlp_manager)
        
        # 클러스터링 엔진 초기화
        logger.info("🎯 클러스터링 엔진 초기화 중...")
        clustering_engine = ClusteringEngine(nlp_manager)
        
        # GPT 라벨러 초기화
        logger.info("🤖 GPT 라벨러 초기화 중...")
        gpt_labeler = GPTLabeler()
        
        logger.info("✅ 모든 NLP 모델 로딩 완료!")
        
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        # 부분적 로딩도 허용 (서비스 지속성)

# 상태 확인 엔드포인트
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """시스템 상태 및 모델 로딩 상태 확인"""
    
    # ConceptNet 연결 테스트
    conceptnet_available = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get("https://api.conceptnet.io/c/ko/테스트")
            conceptnet_available = response.status_code == 200
    except:
        pass
    
    return HealthResponse(
        status="healthy" if nlp_manager else "partial",
        timestamp=datetime.now().isoformat(),
        sbert_loaded=nlp_manager.sbert_model is not None if nlp_manager else False,
        word2vec_loaded=nlp_manager.word2vec_model is not None if nlp_manager else False,
        conceptnet_available=conceptnet_available,
        gpt_available=gpt_labeler.is_available if gpt_labeler else False
    )

# 단어 벡터화 엔드포인트
@app.post("/api/vectorize", response_model=VectorResponse)
async def vectorize_word(request: WordRequest):
    """한국어 단어를 벡터로 변환 (KoSentenceBERT)"""
    
    if not nlp_manager:
        raise HTTPException(status_code=503, detail="NLP 모델이 로딩되지 않음")
    
    try:
        vector, method = await nlp_manager.get_word_vector(request.word)
        
        return VectorResponse(
            word=request.word,
            vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
            method=method
        )
        
    except Exception as e:
        logger.error(f"벡터화 실패 - {request.word}: {e}")
        raise HTTPException(status_code=500, detail=f"벡터화 실패: {str(e)}")

# 의미적 유사도 계산 엔드포인트
@app.post("/api/semantic-similarity", response_model=SimilarityResponse)
async def calculate_semantic_similarity(request: WordPairRequest):
    """두 한국어 단어의 의미적 유사도 계산"""
    
    if not similarity_calculator:
        raise HTTPException(status_code=503, detail="유사도 계산기가 초기화되지 않음")
    
    try:
        similarity, method, confidence = await similarity_calculator.calculate_similarity(
            request.word1, request.word2
        )
        
        return SimilarityResponse(
            word1=request.word1,
            word2=request.word2,
            similarity=similarity,
            method=method,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"유사도 계산 실패 - {request.word1} vs {request.word2}: {e}")
        raise HTTPException(status_code=500, detail=f"유사도 계산 실패: {str(e)}")

# 하이브리드 유사도 계산 엔드포인트
@app.post("/api/hybrid-similarity", response_model=HybridSimilarityResponse)
async def calculate_hybrid_similarity(request: WordPairRequest):
    """다중 방법론을 활용한 하이브리드 유사도 계산"""
    
    if not similarity_calculator:
        raise HTTPException(status_code=503, detail="유사도 계산기가 초기화되지 않음")
    
    try:
        result = await similarity_calculator.calculate_hybrid_similarity(
            request.word1, request.word2
        )
        
        return HybridSimilarityResponse(
            word1=request.word1,
            word2=request.word2,
            hybrid_score=result['hybrid_score'],
            vector_similarity=result['vector_similarity'],
            llm_similarity=result['llm_similarity'],
            conceptnet_score=result['conceptnet_score'],
            confidence=result['confidence']
        )
        
    except Exception as e:
        logger.error(f"하이브리드 유사도 계산 실패 - {request.word1} vs {request.word2}: {e}")
        raise HTTPException(status_code=500, detail=f"하이브리드 유사도 계산 실패: {str(e)}")

# 클러스터링 엔드포인트
@app.post("/api/clustering", response_model=ClusteringResponse)
async def perform_clustering(request: ClusteringRequest):
    """K-means 기반 의미적 클러스터링"""
    
    if not clustering_engine:
        raise HTTPException(status_code=503, detail="클러스터링 엔진이 초기화되지 않음")
    
    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="클러스터링하려면 최소 2개 이상의 단어가 필요")
    
    try:
        # 클러스터 수 자동 조정
        optimal_clusters = min(request.num_clusters, len(request.words) // 2 + 1)
        
        cluster_assignments, silhouette_score, method = await clustering_engine.perform_clustering(
            request.words, optimal_clusters
        )
        
        return ClusteringResponse(
            words=request.words,
            cluster_assignments=cluster_assignments,
            silhouette_score=silhouette_score,
            method=method
        )
        
    except Exception as e:
        logger.error(f"클러스터링 실패 - {request.words}: {e}")
        raise HTTPException(status_code=500, detail=f"클러스터링 실패: {str(e)}")

# GPT 라벨링 엔드포인트
@app.post("/api/generate-labels", response_model=LabelingResponse)
async def generate_cluster_labels(request: LabelingRequest):
    """OpenAI GPT를 활용한 클러스터 라벨 생성"""
    
    if not gpt_labeler:
        raise HTTPException(status_code=503, detail="GPT 라벨러가 초기화되지 않음")
    
    if len(request.words) == 0:
        raise HTTPException(status_code=400, detail="라벨링할 단어가 없음")
    
    try:
        label, confidence = await gpt_labeler.generate_label(request.words)
        
        return LabelingResponse(
            words=request.words,
            label=label,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"GPT 라벨링 실패 - {request.words}: {e}")
        
        # GPT 실패시 기본 라벨 생성
        fallback_label = f"📊 {len(request.words)}개 단어 그룹"
        
        return LabelingResponse(
            words=request.words,
            label=fallback_label,
            confidence=0.3
        )

# ConceptNet 쿼리 엔드포인트
@app.post("/api/conceptnet-query")
async def query_conceptnet(request: WordPairRequest):
    """ConceptNet API를 통한 개념 관계 조회"""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.conceptnet.io/relatedness"
            params = {
                "node1": f"/c/ko/{request.word1}",
                "node2": f"/c/ko/{request.word2}"
            }
            
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "word1": request.word1,
                    "word2": request.word2,
                    "relatedness": data.get("value", 0.0),
                    "source": "conceptnet"
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="ConceptNet API 오류")
                
    except Exception as e:
        logger.error(f"ConceptNet 쿼리 실패 - {request.word1} vs {request.word2}: {e}")
        raise HTTPException(status_code=500, detail=f"ConceptNet 쿼리 실패: {str(e)}")

# 서버 실행 함수
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )
