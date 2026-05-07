import pandas as pd
import numpy as np
import os
import sys

def create_features_pro(df_main, df_stadium):
    """
    KBO 관중수 예측을 위한 전문가급 피처 엔지니어링 함수
    """
    df = df_main.copy()
    
    # [1] 구장명 표준화 및 수용인원 집계 방식 개선 (피드백 1 반영)
    # 한밭->대전 등 명칭 통일 후, 중복 데이터 중 '최대값'을 선택해 왜곡 방지
    df['구장'] = df['구장'].replace({'한밭': '대전', '문학': '인천'}) 
    st_max_table = df_stadium.groupby('구장')['최대수용인원'].max()
    df['stadium_capacity'] = df['구장'].map(st_max_table)
    
    # [2] 매핑 실패(결측) 처리 및 플래그 생성 (피드백 3 반영)
    # 정보가 없는 구장의 경우 0/1 플래그를 생성하여 모델이 인지하게 함
    df['is_capacity_missing'] = df['stadium_capacity'].isnull().astype(int)
    # 결측치는 분석에 방해되지 않도록 전체 평균값으로 대체
    df['stadium_capacity'] = df['stadium_capacity'].fillna(df['stadium_capacity'].mean())

    # [3] 고정 구간(Fixed Cut) 기반 기상 변수 생성 (피드백 2 반영)
    # qcut(분위수) 대신 고정 경계값을 사용하여 학습/추론 시 동일 기준 유지
    df['일합계강수량(mm)'] = df['일합계강수량(mm)'].fillna(0)
    df['is_rain'] = (df['일합계강수량(mm)'] > 0).astype(int)
    
    # 강수량 구간 (None, Light, Medium, Heavy)
    df['rain_bucket'] = pd.cut(df['일합계강수량(mm)'], 
                               bins=[-1, 0, 5, 15, float('inf')], 
                               labels=['None', 'Light', 'Medium', 'Heavy'])
    
    # 기온 구간 (temp_bucket) 및 폭염 여부 (is_hot)
    df['is_hot'] = (df['일평균기온(°C)'] >= 30).astype(int)
    df['temp_bucket'] = pd.cut(df['일평균기온(°C)'], 
                               bins=[-float('inf'), 10, 20, 25, 30, float('inf')], 
                               labels=['VeryCold', 'Cold', 'Mild', 'Warm', 'Hot'])
    
    # 습도 구간 (humidity_bucket) - 일반적 쾌적도 기준 고정 구간
    df['humidity_bucket'] = pd.cut(df['일평균상대습도(%)'], 
                                   bins=[0, 40, 60, 80, 100], 
                                   labels=['Dry', 'Normal', 'Humid', 'VeryHumid'])
    
    # 풍속 구간 (wind_bucket) - 보포트 풍력 계급 응용 고정 구간
    df['wind_bucket'] = pd.cut(df['일평균풍속(m/s)'], 
                                bins=[-1, 1.5, 3.3, 5.4, float('inf')], 
                                labels=['Calm', 'Light', 'Moderate', 'Strong'])

    # [4] 요일 및 주기성 변수 (Cyclical Encoding)
    df['is_weekend'] = df['요일'].isin(['토', '일']).astype(int)
    weekday_map = {'월':0, '화':1, '수':2, '목':3, '금':4, '토':5, '일':6}
    df['weekday_num'] = df['요일'].map(weekday_map)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num'] / 7)
    
    # [5] 2차 상호작용 피처 (피드백 6 반영)
    # 구장별 우천 시 관중 반응이 다를 수 있으므로 조합 변수 생성
    df['stadium_x_rain'] = df['구장'].astype(str) + "_" + df['is_rain'].astype(str)
    
    return df

if __name__ == "__main__":
    # 경로 설정 (명세서 준수)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    
    path_main = os.path.join(project_root, "data/processed/final_dataset.csv")
    path_stadium = os.path.join(project_root, "data/external/kbo_stadium_info.csv")
    save_path = os.path.join(project_root, "data/processed/kbo_train_ready.csv")

    try:
        # 데이터 로드 (파일명 예외 처리)
        if not os.path.exists(path_main):
            path_main = os.path.join(project_root, "data/processed/final_dataset (2).csv")
        
        df_final = pd.read_csv(path_main)
        df_st_info = pd.read_csv(path_stadium) if os.path.exists(path_stadium) else pd.DataFrame()

        # 피처 엔지니어링 실행
        processed_df = create_features_pro(df_final, df_st_info)

        # [6] 학습용 컬럼 리스트 고정 (피드백 4 반영 - 누수 및 실수 방지)
        # 모델 학습에 '직접' 사용할 변수들만 선별하여 데이터셋 크기 최적화 및 안정성 확보
        model_ready_columns = [
            '연도', '월', '주차_ISO', '홈팀', '방문팀', '구장', 
            'stadium_capacity', 'is_capacity_missing',
            'is_rain', 'rain_bucket', 'temp_bucket', 'is_hot',
            'humidity_bucket', 'wind_bucket',
            'is_weekend', 'weekday_sin', 'weekday_cos', 
            'stadium_x_rain',
            '관중수'  # Target 변수
        ]
        
        # 선택된 컬럼만 저장
        final_ready_df = processed_df[model_ready_columns]
        final_ready_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print("Success: Feature engineering for Modeling is completed.")
        print(f"Target Path: {save_path}")
        print(f"Final Column Count: {len(model_ready_columns)}")
        print("-" * 50)

    except Exception as e:
        # 피드백 5 반영: 에러 시 로그를 남기고 시스템 비정상 종료를 알려 파이프라인 중단
        print(f"FATAL ERROR during build_features.py: {str(e)}")
        sys.exit(1)