import streamlit as st
import openpyxl
import pandas as pd
data_file = "./final_100_hr_kings.xlsx"
def main():
    st.title("Best HR Hitter 2024")
    # 선수 이름 입력 창
    search_type = '선수 선택'
    if search_type == '선수 이름 검색':
        player_name = st.text_input('선수 이름을 입력하세요:')
    else:
        df = pd.read_excel(data_file)
        player_names = df['PLAYER'].tolist()
        player_name = st.selectbox('선수 선택', player_names)

    # 검색 버튼
    if st.button('Search'):
        if search_type == '선수 이름 검색':
            hr = find_hr_by_player(player_name)
            st.write(f"{player_name}의 홈런 개수: {hr}")
        else:
            df = pd.read_excel(data_file)
            player_row = df[df['PLAYER'] == player_name]
            if len(player_row) == 0:
                st.write("Not Found")
            else:
                hr = player_row.iloc[0]['HR']
                st.write(f"{player_name}의 홈런 개수: {hr}")
    # prediction
    if st.button("Top 20 Player"):
        prediction_df = pd.read_excel('./final_20_hr_kings.xlsx')
        prediction_df.columns = ["RANK", "PLAYER", "HR"]
        # prediction_df.drop("RANK", inplace = True)
        prediction_df.index = prediction_df.index + 1
        st.write(prediction_df)
if __name__ == '__main__':
    main()

