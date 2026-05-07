import pandas as pd
import numpy as np
import os
import sys

def create_features_pro(df_main, df_stadium):
    df = df_main.copy()
    
    # [1] 구장명 표준화 및 수용인원 MAX 집계 (왜곡 방지)
    df['구장'] = df['구장'].replace({'한밭': '대전', '문학': '인천'}) 
    st_max_table = df_stadium.groupby('구장')['최대수용인원'].max()
    df['stadium_capacity'] = df['구장'].map(st_max_table)
    
    # [2] 결측 처리 및 플래그 생성
    df['is_capacity_missing'] = df['stadium_capacity'].isnull().astype(int)
    df['stadium_capacity'] = df['stadium_capacity'].fillna(df['stadium_capacity'].mean())

    # [3] 고정 구간 기반 기상 변수 (라벨 명확화: None -> No_Rain)
    df['일합계강수량(mm)'] = df['일합계강수량(mm)'].fillna(0)
    df['is_rain'] = (df['일합계강수량(mm)'] > 0).astype(int)
    
    # rain_bucket 라벨에서 'None'을 'No_Rain'으로 수정하여 결측치 오인 방지
    df['rain_bucket'] = pd.cut(df['일합계강수량(mm)'], 
                               bins=[-1, 0, 5, 15, float('inf')], 
                               labels=['No_Rain', 'Light', 'Medium', 'Heavy'])
    
    # 기온 구간
    df['is_hot'] = (df['일평균기온(°C)'] >= 30).astype(int)
    df['temp_bucket'] = pd.cut(df['일평균기온(°C)'], 
                               bins=[-float('inf'), 10, 20, 25, 30, float('inf')], 
                               labels=['VeryCold', 'Cold', 'Mild', 'Warm', 'Hot'])
    
    # 습도/풍속 고정 구간
    df['humidity_bucket'] = pd.cut(df['일평균상대습도(%)'], 
                                   bins=[0, 40, 60, 80, 100], 
                                   labels=['Dry', 'Normal', 'Humid', 'VeryHumid'])
    df['wind_bucket'] = pd.cut(df['일평균풍속(m/s)'], 
                                bins=[-1, 1.5, 3.3, 5.4, float('inf')], 
                                labels=['Calm', 'Light', 'Moderate', 'Strong'])

    # [4] 요일 및 주기성 변수
    df['is_weekend'] = df['요일'].isin(['토', '일']).astype(int)
    weekday_map = {'월':0, '화':1, '수':2, '목':3, '금':4, '토':5, '일':6}
    df['weekday_num'] = df['요일'].map(weekday_map)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num'] / 7)
    
    # [5] 상호작용 변수
    df['stadium_x_rain'] = df['구장'].astype(str) + "_" + df['is_rain'].astype(str)
    
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    
    path_main = os.path.join(project_root, "data/processed/final_dataset.csv")
    path_stadium = os.path.join(project_root, "data/external/kbo_stadium_info.csv")
    save_path = os.path.join(project_root, "data/processed/kbo_train_ready.csv")

    try:
        if not os.path.exists(path_main):
            path_main = os.path.join(project_root, "data/processed/final_dataset (2).csv")
        
        df_final = pd.read_csv(path_main)
        df_st_info = pd.read_csv(path_stadium) if os.path.exists(path_stadium) else pd.DataFrame()

        processed_df = create_features_pro(df_final, df_st_info)

        # [6] 최종 학습용 스키마 (Target Leakage 방지 및 변수 선별)
        model_ready_columns = [
            '연도', '월', '주차_ISO', '홈팀', '방문팀', '구장', 
            'stadium_capacity', 'is_capacity_missing',
            'is_rain', 'rain_bucket', 'temp_bucket', 'is_hot',
            'humidity_bucket', 'wind_bucket',
            'is_weekend', 'weekday_sin', 'weekday_cos', 
            'stadium_x_rain',
            '관중수' # Target
        ]
        
        final_ready_df = processed_df[model_ready_columns]
        final_ready_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print("Success: Final Data for ML pipeline is generated.")
        print(f"Location: {save_path}")
        print("Note: rain_bucket 'None' is renamed to 'No_Rain' for safety.")
        print("-" * 50)

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        sys.exit(1)