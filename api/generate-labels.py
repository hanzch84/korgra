from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse

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
            
            if not words:
                raise ValueError("단어 목록이 비어있습니다")
            
            # OpenAI API 시도
            openai_label = self.generate_openai_label(words)
            
            if openai_label:
                response = {
                    "label": openai_label,
                    "method": "openai-gpt",
                    "confidence": "high"
                }
            else:
                # OpenAI 실패시 패턴 기반 라벨링
                pattern_label = self.generate_pattern_label(words)
                response = {
                    "label": pattern_label,
                    "method": "pattern-based",
                    "confidence": "medium"
                }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            print(f"라벨링 오류: {str(e)}")
            # 에러 발생시 기본 라벨
            fallback_label = f"📊 그룹 ({len(data.get('words', []))}개 단어)"
            response = {
                "label": fallback_label,
                "method": "fallback",
                "confidence": "low",
                "error": str(e)
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def generate_openai_label(self, words):
        """OpenAI GPT를 사용하여 클러스터 라벨 생성"""
        try:
            # 환경변수에서 API 키 가져오기
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("OpenAI API 키가 설정되지 않음")
                return None
            
            # 단어 목록을 문자열로 변환
            words_str = ', '.join(words)
            
            # GPT 프롬프트 설계
            prompt = f"""다음 한국어 단어들의 공통 주제나 카테고리를 분석하고, 간결하고 직관적인 그룹 이름을 생성하세요.

단어 목록: {words_str}

요구사항:
1. 이모지 1개 + 한국어 2-4글자로 구성
2. 교육적이고 이해하기 쉬운 표현
3. 단어들의 핵심 공통점을 반영
4. 예시 형식: "🐾 동물", "🌟 천문학", "🎨 색깔"

그룹 이름:"""

            # OpenAI API 요청 데이터
            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system", 
                        "content": "당신은 한국어 교육 전문가입니다. 단어들을 분석하여 의미있는 카테고리 이름을 생성하는 것이 전문 분야입니다."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.3,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            # HTTP 요청 준비
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 요청 데이터를 JSON으로 인코딩
            post_data = json.dumps(request_data).encode('utf-8')
            
            # HTTP 요청 생성
            req = urllib.request.Request(url, data=post_data, headers=headers, method='POST')
            
            # API 호출 (5초 타임아웃)
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                # 응답에서 생성된 라벨 추출
                if 'choices' in result and len(result['choices']) > 0:
                    label = result['choices'][0]['message']['content'].strip()
                    
                    # 라벨 검증 및 정제
                    label = self.validate_and_clean_label(label)
                    
                    print(f"OpenAI 라벨 생성 성공: {label}")
                    return label
                else:
                    print("OpenAI 응답에 유효한 content가 없음")
                    return None
                    
        except urllib.error.HTTPError as e:
            print(f"OpenAI API HTTP 오류: {e.code} - {e.reason}")
            return None
        except urllib.error.URLError as e:
            print(f"OpenAI API 연결 오류: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"OpenAI API 응답 파싱 오류: {e}")
            return None
        except Exception as e:
            print(f"OpenAI API 알 수 없는 오류: {e}")
            return None
    
    def validate_and_clean_label(self, label):
        """생성된 라벨 검증 및 정제"""
        if not label:
            return None
        
        # 불필요한 따옴표나 줄바꿈 제거
        label = label.strip().strip('"\'').strip()
        
        # 최대 길이 제한 (20자)
        if len(label) > 20:
            label = label[:20] + "..."
        
        # 기본 이모지가 없으면 추가
        if not any(ord(char) > 127 for char in label[:2]):  # 이모지 감지
            label = f"📊 {label}"
        
        return label
    
    def generate_pattern_label(self, words):
        """패턴 기반 자동 라벨링 (OpenAI 대안)"""
        categories = {
            'animals': {
                'keywords': ['강아지', '고양이', '토끼', '새', '물고기', '개', '닭', '말', '소', '돼지', '사자', '호랑이', '곰', '늑대'],
                'label': '🐾 동물'
            },
            'space': {
                'keywords': ['노바', '초신성', '별', '우주', '은하', '행성', '태양', '달', '화성', '지구', '성운', '혜성'],
                'label': '🌟 천문학'
            },
            'colors': {
                'keywords': ['빨간색', '파란색', '노란색', '초록색', '검은색', '하얀색', '색깔', '컬러', '빨강', '파랑', '노랑', '초록'],
                'label': '🎨 색깔'
            },
            'family': {
                'keywords': ['아빠', '엄마', '형', '누나', '동생', '가족', '부모', '형제', '자매', '할머니', '할아버지'],
                'label': '👨‍👩‍👧‍👦 가족'
            },
            'food': {
                'keywords': ['밥', '빵', '음식', '치킨', '피자', '국수', '라면', '김치', '과일', '야채', '고기'],
                'label': '🍽️ 음식'
            },
            'sports': {
                'keywords': ['축구', '야구', '농구', '테니스', '수영', '운동', '스포츠', '경기', '선수'],
                'label': '⚽ 스포츠'
            },
            'school': {
                'keywords': ['학교', '선생님', '학생', '교실', '공부', '책', '연필', '시험', '숙제'],
                'label': '🎓 학교'
            },
            'emotions': {
                'keywords': ['기쁨', '슬픔', '화남', '사랑', '행복', '감정', '마음', '느낌'],
                'label': '💝 감정'
            },
            'nature': {
                'keywords': ['나무', '꽃', '산', '바다', '강', '숲', '자연', '환경'],
                'label': '🌿 자연'
            },
            'technology': {
                'keywords': ['컴퓨터', '스마트폰', '인터넷', '기술', '로봇', 'AI', '프로그램'],
                'label': '💻 기술'
            }
        }
        
        # 단어 매칭으로 카테고리 찾기
        for category, data in categories.items():
            matched_count = sum(1 for word in words if word in data['keywords'])
            
            # 50% 이상 매칭되면 해당 카테고리
            if matched_count >= len(words) * 0.5:
                return data['label']
        
        # 매칭되는 카테고리가 없으면 단어 기반 라벨
        if len(words) == 1:
            return f"🔸 {words[0]}"
        elif len(words) <= 3:
            return f"📝 {words[0]}관련"
        else:
            return f"📊 혼합그룹 ({len(words)}개)"
