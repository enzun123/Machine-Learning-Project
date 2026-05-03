import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

def scrape_kbo_attendance(years):
    chrome_options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    url = "https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx"
    driver.get(url)
    time.sleep(3) 
    
    all_data = {year: [] for year in years}

    try:
        for year in years:
            print(f"--- {year}년 데이터 수집 시작 ---")
            
            try:
                year_dropdown = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlSeason']") 
                year_select = Select(year_dropdown)
                year_select.select_by_value(str(year))
                time.sleep(2) 
            except Exception as e:
                print(f"연도 선택 중 오류 (사이트 구조 확인 필요): {e}")
                continue

            page_num = 1
            while True:
                print(f"{year}년 - {page_num}페이지 수집 중...")
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                table = soup.find('table', {'class': 'tData'})
                if not table:
                    table = soup.find('table', {'class': 'tbl'})
                    if not table:
                        print("데이터 표(Table)를 찾을 수 없습니다. 수집 종료.")
                        break
                    
                rows = table.find('tbody').find_all('tr')
                
                for row in rows:
                    cols = row.find_all('td')
                    
                    if len(cols) < 6 or "없습니다" in cols[0].text:
                        continue
                        
                    row_data = [col.text.strip() for col in cols]
                    
                    crowd_text = row_data[-1] 
                    if crowd_text == '-' or crowd_text == '0':
                        continue 
                        
                    all_data[year].append(row_data)

                try:
                    # KBO 페이지네이션 구조: 현재 페이지 번호의 다음 요소를 찾습니다.
                    # '다음' 버튼(> 이미지)을 명시적으로 찾도록 수정
                    next_button = driver.find_element(By.CSS_SELECTOR, "a.next")
                    
                    # 버튼의 href 속성이나 클래스를 검사하여 마지막 페이지인지 확인
                    if "disabled" in next_button.get_attribute("class") or next_button.get_attribute("href") in [None, "", "javascript:void(0);"]:
                        print(f"{year}년 마지막 페이지 도달.")
                        break 
                        
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2) 
                    page_num += 1
                except Exception:
                    print(f"{year}년 모든 페이지 수집 완료.")
                    break
                    
    except Exception as e:
        print(f"스크래핑 중 치명적인 오류 발생: {e}")
    finally:
        driver.quit() 

    return all_data

if __name__ == "__main__":
    target_years = [2024, 2025]
    
    scraped_data_dict = scrape_kbo_attendance(target_years)
    
    # 💡 수정 포인트: 컬럼 개수를 실제 표 구조인 6개로 맞춤
    column_names = ['날짜', '요일', '시간', '경기(원정vs홈)', '구장', '관중수'] 
    
    for year, data in scraped_data_dict.items():
        if not data:
            print(f"\n{year}년 수집된 데이터가 없습니다.")
            continue
            
        df = pd.DataFrame(data, columns=column_names)
        
        df['관중수'] = df['관중수'].str.replace(',', '').astype(int)
        
        if 'vs' in str(df['경기(원정vs홈)'].iloc[0]):
            df[['방문', '홈']] = df['경기(원정vs홈)'].str.split('vs', expand=True)
            df['방문'] = df['방문'].str.strip()
            df['홈'] = df['홈'].str.strip()
            # 시간과 원본 경기 컬럼 삭제
            df = df.drop(columns=['경기(원정vs홈)', '시간']) 
            
            # 머신러닝에 편하도록 열 순서 재배치
            df = df[['날짜', '요일', '홈', '방문', '구장', '관중수']]
        
        print(f"\n--- {year}년 최종 데이터 ---")
        print(df.head())
        print(f"총 {len(df)}건의 데이터 수집 완료!")
        
        file_name = f"kbo_{year}_attendance_real.csv"
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"{file_name} 파일이 성공적으로 저장되었습니다.")