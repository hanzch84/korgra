from http.server import BaseHTTPRequestHandler
import json
import os

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
            
            words = data.get('words', [])
            
            # 간단한 패턴 기반 라벨링 (OpenAI 대신)
            label = self.generate_pattern_label(words)
            
            response = {
                "label": label,
                "method": "pattern-based",
                "confidence": "medium"
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
    
    def generate_pattern_label(self, words):
        # 패턴 기반 자동 라벨링
        categories = {
            'animals': (['강아지', '고양이', '토끼', '새', '물고기', '개', '닭'], '🐾 동물'),
            'space': (['노바', '초신성', '별', '우주', '은하', '행성'], '🌟 천문학'),
            'colors': (['빨간색', '파란색', '노란색', '초록색', '색깔'], '🎨 색깔'),
            'family': (['아빠', '엄마', '형', '누나', '동생', '가족'], '👨‍👩‍👧‍👦 가족'),
            'food': (['밥', '빵', '음식', '치킨', '피자'], '🍽️ 음식'),
            'sports': (['축구', '야구', '농구', '운동'], '⚽ 스포츠')
        }
        
        for category, (keywords, label) in categories.items():
            if any(word in keywords for word in words):
                return label
        
        # 기본 라벨
        return f"📊 그룹 ({len(words)}개 단어)"
