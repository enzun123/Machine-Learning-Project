import pandas as pd
import numpy as np
import os

def create_features_final(df_main, df_stadium):
    """
    KBO 관중수 예측을 위한 최종 피처 엔지니어링 함수
    - 명세서 [4]번 항목의 모든 구간화(Bucket) 변수 포함
    """
    df = df_main.copy()
    
    # [1] 구장명 표준화 (한밭 -> 대전, 문학 -> 인천)
    df['구장'] = df['구장'].replace({'한밭': '대전', '문학': '인천'}) 
    
    # [2] 구장 수용인원 조인
    if not df_stadium.empty:
        # 중복 구장 제거 (잠실 등) 후 최대수용인원 매핑
        st_map = df_stadium.drop_duplicates('구장').set_index('구장')['최대수용인원']
        df['stadium_capacity'] = df['구장'].map(st_map)
    else:
        df['stadium_capacity'] = np.nan

    # [3] 기상 데이터 구간화 (Bucket) 변수 생성 - 명세서 반영
    # 3-1. 강수량 (is_rain, rain_bucket)
    df['일합계강수량(mm)'] = df['일합계강수량(mm)'].fillna(0)
    df['is_rain'] = (df['일합계강수량(mm)'] > 0).astype(int)
    df['rain_bucket'] = pd.cut(df['일합계강수량(mm)'], 
                               bins=[-1, 0, 1, 5, float('inf')], 
                               labels=['0', '0-1', '1-5', '5+'])

    # 3-2. 기온 (temp_bucket, is_hot)
    df['is_hot'] = (df['일평균기온(°C)'] >= 30).astype(int)
    df['temp_bucket'] = pd.cut(df['일평균기온(°C)'], 
                               bins=[-float('inf'), 10, 20, 25, 30, float('inf')], 
                               labels=['<10', '10-20', '20-25', '25-30', '30+'])

    # 3-3. 습도 및 풍속 (humidity_bucket, wind_bucket) - 4분위수 기준 구간화
    df['humidity_bucket'] = pd.qcut(df['일평균상대습도(%)'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['wind_bucket'] = pd.qcut(df['일평균풍속(m/s)'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # [4] 요일 및 주기성 변수 (is_weekend, weekday_sin/cos)
    df['is_weekend'] = df['요일'].isin(['토', '일']).astype(int)
    weekday_map = {'월':0, '화':1, '수':2, '목':3, '금':4, '토':5, '일':6}
    df['weekday_num'] = df['요일'].map(weekday_map)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num'] / 7)
    
    # [5] 2차 확장 변수 (상호작용 변수 예시)
    df['weekend_x_rain'] = df['is_weekend'].astype(str) + "_" + df['is_rain'].astype(str)
    
    return df

if __name__ == "__main__":
    # 1. 경로 설정 (프로젝트 구조 준수)
    # 현재 파일 위치: scripts/features/build_features.py 기준
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../")) # 프로젝트 최상위로 이동

    # 입력 파일 경로
    path_main = os.path.join(project_root, "data", "processed", "final_dataset.csv")
    path_stadium = os.path.join(project_root, "data", "external", "kbo_stadium_info.csv")
    
    # 출력 파일 경로 (파일명: kbo_train_ready.csv)
    save_path = os.path.join(project_root, "data", "processed", "kbo_train_ready.csv")

    try:
        # 데이터 로드 시도 (파일명 예외 처리 포함)
        if not os.path.exists(path_main):
            path_main = os.path.join(project_root, "data", "processed", "final_dataset (2).csv")
            
        print(f"Loading data from: {path_main}")
        df_final = pd.read_csv(path_main)
        
        if os.path.exists(path_stadium):
            df_st_info = pd.read_csv(path_stadium)
            print("Success: Stadium info loaded.")
        else:
            print("Warning: Stadium info file not found. Capacity will be NaN.")
            df_st_info = pd.DataFrame()

        # 2. 피처 엔지니어링 실행
        final_df = create_features_final(df_final, df_st_info)
        
        # 3. 최종 CSV 저장 (utf-8-sig로 인코딩하여 엑셀 한글 깨짐 방지)
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print("Success: Feature engineering completed.")
        print(f"File Saved: {save_path}")
        print("-" * 50)

    except Exception as e:
        print(f"Error occurred: {str(e)}")