import pandas as pd
import numpy as np
import os

def create_features_final(df_main, df_stadium):
    df = df_main.copy()
    # [1] 구장명 표준화[cite: 1]
    df['구장'] = df['구장'].replace({'한밭': '대전', '문학': '인천'}) 
    
    # [2] 구장 수용인원 조인[cite: 1]
    if not df_stadium.empty:
        st_map = df_stadium.drop_duplicates('구장').set_index('구장')['최대수용인원']
        df['stadium_capacity'] = df['구장'].map(st_map)
    else:
        df['stadium_capacity'] = np.nan

    # [3] 강수/기온/요일 파생변수 생성[cite: 1]
    df['일합계강수량(mm)'] = df['일합계강수량(mm)'].fillna(0)
    df['is_rain'] = (df['일합계강수량(mm)'] > 0).astype(int)
    df['rain_bucket'] = pd.cut(df['일합계강수량(mm)'], bins=[-1, 0, 1, 5, float('inf')], labels=['0', '0-1', '1-5', '5+'])
    df['is_weekend'] = df['요일'].isin(['토', '일']).astype(int)
    weekday_map = {'월':0, '화':1, '수':2, '목':3, '금':4, '토':5, '일':6}
    df['weekday_num'] = df['요일'].map(weekday_map)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num'] / 7)
    df['is_hot'] = (df['일평균기온(°C)'] >= 30).astype(int)
    
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_main = os.path.join(current_dir, "final_dataset (2).csv")
    path_stadium = os.path.join(current_dir, "kbo_stadium_info.csv")

    try:
        # 데이터 로드
        df_final = pd.read_csv(path_main)
        
        if os.path.exists(path_stadium):
            df_st_info = pd.read_csv(path_stadium)
            print("Success: Stadium info loaded.")
        else:
            print("Warning: Stadium info file not found.")
            df_st_info = pd.DataFrame()

        # 파생변수 생성
        final_df = create_features_final(df_final, df_st_info)
        
        # 💾 [핵심] CSV 파일로 저장
        # 한글 깨짐 방지를 위해 'utf-8-sig' 인코딩 사용
        save_path = os.path.join(current_dir, "featured_kbo_data.csv")
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 30)
        print(f"Success: Final file saved at -> {save_path}")
        print("-" * 30)

    except Exception as e:
        print(f"Error: {str(e)}")