import pandas as pd  # 이 줄이 반드시 맨 위에 있어야 합니다!

# 구장별 수용 인원 기준표
stadium_info = {
    '구단': ['LG', '두산', '삼성', 'SSG', '롯데', 'NC', 'KIA', '한화', 'KT', '키움'],
    '구장': ['잠실', '잠실', '대구', '인천', '부산', '창원', '광주', '대전', '수원', '고척'],
    '최대수용인원': [23750, 23750, 24000, 23000, 22990, 17891, 20500, 20000, 18700, 16000]
}

# 이제 pd를 사용할 수 있습니다.
df_stadium = pd.DataFrame(stadium_info)

# 파일 저장까지 하려면
df_stadium.to_csv("kbo_stadium_info.csv", index=False, encoding="utf-8-sig")

print("구장 정보 파일 생성 완료!")