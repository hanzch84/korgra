from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request

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
            
            word = data.get('word', '')
            
            if not word:
                raise ValueError("단어가 제공되지 않았습니다")
            
            # OpenAI 임베딩 생성
            vector = self.generate_embedding(word)
            
            response = {
                "vector": vector,
                "dimensions": len(vector) if vector else 0,
                "model": "text-embedding-ada-002"
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            print(f"벡터화 오류: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def generate_embedding(self, word):
        """OpenAI 임베딩 생성"""
        try:
            # 환경변수에서 API 키 가져오기
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API 키가 설정되지 않음")
            
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            request_data = {
                "model": "text-embedding-ada-002",
                "input": word,
                "encoding_format": "float"
            }
            
            post_data = json.dumps(request_data).encode('utf-8')
            req = urllib.request.Request(url, data=post_data, headers=headers, method='POST')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    print(f"임베딩 벡터화 성공: {word} → {len(embedding)}차원")
                    return embedding
                else:
                    raise ValueError("임베딩 응답에 데이터 없음")
                    
        except Exception as e:
            print(f"임베딩 생성 실패: {word} - {e}")
            raise e
