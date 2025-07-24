from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse
import math

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # CORS 헤더 설정
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # 요청 데이터 읽기
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            word1 = data.get('word1', '')
            word2 = data.get('word2', '')
            
            if not word1 or not word2:
                raise ValueError("단어가 제공되지 않았습니다")
            
            # OpenAI 임베딩으로 유사도 계산
            similarity = self.calculate_embedding_similarity(word1, word2)
            
            response = {
                "similarity": similarity,
                "method": "openai-embedding-ada-002",
                "confidence": "high" if similarity > 0.7 else "medium"
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            print(f"임베딩 유사도 계산 오류: {str(e)}")
            # 실패시 ConceptNet 백업
            similarity = self.get_conceptnet_similarity(word1, word2)
            response = {
                "similarity": similarity,
                "method": "conceptnet-backup",
                "confidence": "medium",
                "error": str(e)
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def calculate_embedding_similarity(self, word1, word2):
        """OpenAI 임베딩으로 의미적 유사도 계산"""
        try:
            # 환경변수에서 API 키 가져오기
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("OpenAI API 키가 설정되지 않음")
                return self.get_conceptnet_similarity(word1, word2)
            
            # 두 단어의 임베딩 벡터 가져오기
            embedding1 = self.get_embedding(word1, api_key)
            embedding2 = self.get_embedding(word2, api_key)
            
            if embedding1 and embedding2:
                # 코사인 유사도 계산
                similarity = self.cosine_similarity(embedding1, embedding2)
                print(f"임베딩 유사도: {word1} ↔ {word2} = {similarity:.3f}")
                return similarity
            else:
                # 임베딩 실패시 ConceptNet 백업
                return self.get_conceptnet_similarity(word1, word2)
                
        except Exception as e:
            print(f"임베딩 계산 실패: {e}")
            return self.get_conceptnet_similarity(word1, word2)
    
    def get_embedding(self, text, api_key):
        """OpenAI 임베딩 API 호출"""
        try:
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            request_data = {
                "model": "text-embedding-ada-002",
                "input": text,
                "encoding_format": "float"
            }
            
            post_data = json.dumps(request_data).encode('utf-8')
            req = urllib.request.Request(url, data=post_data, headers=headers, method='POST')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    print(f"임베딩 성공: {text} → {len(embedding)}차원")
                    return embedding
                else:
                    print(f"임베딩 실패: {text} - 응답에 데이터 없음")
                    return None
                    
        except Exception as e:
            print(f"임베딩 API 호출 실패: {text} - {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """코사인 유사도 계산"""
        try:
            # 내적 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # 벡터 크기 계산
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            # 코사인 유사도
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            
            # 0-1 범위로 정규화 (임베딩은 -1~1 범위)
            normalized_similarity = (similarity + 1) / 2
            
            return max(0.0, min(1.0, normalized_similarity))
            
        except Exception as e:
            print(f"코사인 유사도 계산 오류: {e}")
            return 0.0
    
    def get_conceptnet_similarity(self, word1, word2):
        """ConceptNet 백업 유사도 (임베딩 실패시)"""
        try:
            url = f"https://api.conceptnet.io/relatedness?node1=/c/ko/{urllib.parse.quote(word1)}&node2=/c/ko/{urllib.parse.quote(word2)}"
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                similarity = data.get('value', 0.0)
                return max(0.1, min(0.95, similarity))
        except:
            # 모든 방법 실패시 기본 패턴 매칭
            return self.calculate_basic_similarity(word1, word2)
    
    def calculate_basic_similarity(self, word1, word2):
        """기본 패턴 매칭 (최후 백업)"""
        patterns = {
            'animals': ['강아지', '고양이', '토끼', '새', '물고기'],
            'space': ['노바', '초신성', '별', '우주', '은하'],
            'colors': ['빨간색', '파란색', '노란색', '초록색'],
            'family': ['아빠', '엄마', '형', '누나', '동생']
        }
        
        for category, words in patterns.items():
            if word1 in words and word2 in words:
                return 0.8
        
        return 0.2
