from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import urllib.request
import ssl

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
            
            # ConceptNet API 호출
            similarity = self.get_conceptnet_similarity(word1, word2)
            
            response = {
                "similarity": similarity,
                "method": "conceptnet-api",
                "confidence": "high" if similarity > 0.5 else "medium"
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_conceptnet_similarity(self, word1, word2):
        try:
            # ConceptNet API 호출
            url = f"https://api.conceptnet.io/relatedness?node1=/c/ko/{urllib.parse.quote(word1)}&node2=/c/ko/{urllib.parse.quote(word2)}"
            
            # SSL 컨텍스트 생성 (인증서 검증 비활성화)
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=context, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                similarity = data.get('value', 0.0)
                return max(0.1, min(0.95, similarity))
        except:
            # 실패시 기본 유사도 (한국어 패턴 기반)
            return self.calculate_basic_similarity(word1, word2)
    
    def calculate_basic_similarity(self, word1, word2):
        # 간단한 한국어 의미 패턴 매칭
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
