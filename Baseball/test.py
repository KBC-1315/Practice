def prediction() :
    # 필요한 라이브러리 로드
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    hitter_2022 = pd.read_csv(".\hitter_2022.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2023 = pd.read_csv(".\hitter_2023.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2024 = pd.read_csv(".\hitter_2024.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_total = pd.read_csv(".\hitter_total.csv", encoding = "UTF-8", index_col = "RANK")

    hitter_2022 = hitter_2022.drop("Unnamed: 0", axis = 1)
    hitter_2023 = hitter_2023.drop("Unnamed: 0", axis = 1)
    hitter_2024 = hitter_2024.drop("Unnamed: 0", axis = 1)
    hitter_total = hitter_total.drop("Unnamed: 0", axis = 1)

    # hitter_total df의 컬럼 순서가 바뀌어 있어서 이름 재정의
    hitter_total.columns = ["PLAYER", "POSITION", "YEAR", "TEAM", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS"]

    # hitter_total df의 YEAR 컬럼은 사용하지 않을것 -> 2024년의 HR 데이터를 예측할 것이기 때문! -> 컬럼 제거
    hitter_total = hitter_total.drop("YEAR", axis = 1)

    # \u200c 행은 날려주자
    hitter_total = hitter_total.drop(hitter_total[hitter_total["POSITION"] == "\u200c"].index, axis = 0)

    # SO 컬럼에 결측치가 있다...? -> 찾아보니 STRIKE OUT이더라 아마 삼진아웃이 없지 않았을까?? -> 결측값을 0으로 채워주자 -> 다른컬럼에는 결측치가 없으니 전체적으로 채워주자
    hitter_total.fillna(0, inplace = True)

    #  값이 없는것에 "--" 표시를 해놓은듯...? -> 0으로 변경해보자 -> Stolen Base 라는 뜻이래요
    hitter_total.loc[hitter_total["SB"] == "--", "SB"] = 0

    #  값이 없는것에 "--" 표시를 해놓은듯...? -> 0으로 변경해보자
    hitter_total.loc[hitter_total["CS"] == "--", "CS"] = 0

    # 전부다 object 타입...? preprocessing이 필요할듯 싶다
    # 먼저 누가봐도 String 타입인 친구들 - "POSITION", "TEAM" 부터 변환을 해주자
    from sklearn.preprocessing import LabelEncoder

    col_list = ["POSITION", "TEAM"]
    temp = pd.DataFrame()
    for i in col_list :
        label_encoder = LabelEncoder()
        label_encoder.fit(hitter_total[i])
        hitter_2023[i] = label_encoder.transform(hitter_2023[i])
        hitter_2022[i] = label_encoder.transform(hitter_2022[i])
        hitter_2024[i] = label_encoder.transform(hitter_2024[i])
        hitter_total[i] = label_encoder.transform(hitter_total[i])
    #hitter_total.info()

    # 예쁘게 정수형으로 변환이 되었으니 밑에 숫자인척하는 object들을 변환해주자 뒤에 AVG, OBP, SLG, OPS은 실수형, 이거 빼고는 정수형으로 변환해주면 될것같다
    # 정수형으로 변환을 먼저 해보자
    col_list = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS"]

    for i in col_list :
        hitter_total[i] = hitter_total[i].astype("int64")
        print(i,"컬럼을 정수형으로 변환")
    #hitter_total.dtypes

    # 이제 AVG, OBP, SLG, OPS를 실수형을 바꾸려고 하는데 .000 형태로 되어있어 바로 바뀌지는 않을듯...? -> 이게 되네
    col_list = ["AVG", "OBP", "SLG", "OPS"]

    for i in col_list :
        hitter_total[i] = hitter_total[i].astype("float")
        print(i,"컬럼을 실수형으로 변환")
    #hitter_total.dtypes

    # VIF가 10 이상인것 즉 다중공선성이 확인되는 컬럼들만 따로 하나씩 확인해보자 G, AB, R, H, 2B, 3B, HR, RBI, BB, SO, SB, CS, AVG, OBP, SLG, OPS
    # 이중에서 상관계수의 값이 0.5이상일시에 다중공선성이 있는 컬럼으로 최종 판단하여 교차항을 생성하자
    col_names = ["G", "AB", "R", "H", "2B", "3B", "RBI", "BB", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS"]
    temp_no = 0
    temp_hitter_total = pd.DataFrame()
    for j in col_names :
        temp = hitter_total.corr()[j] > 0.5
        n = 0
        for i in temp.index :
            if i != "HR" :
                if (temp.iloc[n] == True) and (j != i) :
                    if j+"*"+i in temp_hitter_total.columns :
                        print(j," * ",i,"가 이미 있어서 스킵")
                    elif i+"*"+j in temp_hitter_total.columns :
                        print(j," * ",i,"가 이미 있어서 스킵")
                    else:
                        print(j, "*", i)
                        temp_no += 1
                        temp_hitter_total[j+"*"+i] = hitter_total[j] * hitter_total[i]
                n += 1
    print("교차항 :", temp_no,"개 필요함")

    # 상관계수 절댓값이 0.5가 넘은 최후의 합격자들 14개 특성
    temp_corr = temp_corr.drop(temp_corr[temp_corr == False].index, axis = 0)
    # temp_corr

    # 최후의 합격자들로만 데이터프레임 구성
    result_df = pd.DataFrame()
    col_names_1 = ["3B", "HR", "RBI", "SO", "SLG"]
    col_names_2 = ["RBI*G", "RBI*AB", "RBI*BB", "RBI*OPS", "BB*SO", "SO*RBI", "OBP*SO", "SLG*RBI", "SLG*OPS"]

    for i in col_names_1 :
        result_df = pd.concat([result_df,hitter_total[i]], axis = 1)
    for j in col_names_2 :
        result_df = pd.concat([result_df,temp_hitter_total[j]], axis = 1)
    # result_df

    # 최종 데이터의 왜곡 정도를 확인 및 왜곡 정도가 높은 친구들을 로그 스케일로 변환
    from scipy.stats import skew

    features_index = result_df.dtypes[result_df.dtypes != "object"].index

    skew_features = result_df[features_index].apply(lambda x : skew(x))

    skew_features_top = skew_features[skew_features > 1]
    print(skew_features_top.sort_values(ascending = False))

    # 4개 친구들이 왜도가 높다 => 로그 스케일로 변환
    result_df[skew_features_top.index] = np.log1p(result_df[skew_features_top.index])

    # 다시 데이터의 왜곡 정도를 확인

    features_index = result_df.dtypes[result_df.dtypes != "object"].index

    skew_features = result_df[features_index].apply(lambda x : skew(x))

    skew_features_top = skew_features[skew_features > 1]
    print(skew_features_top.sort_values(ascending = False))

    X = result_df.drop(["HR"], axis = 1)
    y = result_df["HR"]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    # 검증셋 크기는 0.3 정도로
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 1004)
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    # Robust 스케일러 활용 스케일링 진행
    rb_scaler = RobustScaler()
    X_train_rb = rb_scaler.fit_transform(X_train)
    X_valid_rb = rb_scaler.transform(X_valid)
    X_train_rb = pd.DataFrame(X_train_rb, index = X_train.index, columns = X_train.columns)
    X_train_rb
    rb_scaler_y = RobustScaler()
    #y_train = y_train.values.reshape(-1, 1)
    #y_valid = y_valid.values.reshape(-1, 1)
    #y_train = rb_scaler_y.fit_transform(y_train)
    #y_valid = rb_scaler_y.transform(y_valid)
    print(y_train.shape)
    print(y_valid.shape)

    # 일단 한번 학습을 시켜보자
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from xgboost import XGBRegressor
    import math
    pd.options.display.float_format = '{:.4f}'.format

    result_matrix = pd.DataFrame()
    # 선형회귀
    lg_model = LinearRegression()
    lg_model.fit(X_train_rb, y_train)
    lg_result = {"Model" : "LinearRegression"}
    lg_result["Train score"] = lg_model.score(X_train, y_train)
    lg_result["Test score"] = lg_model.score(X_valid, y_valid)
    lg_result["Train RMSE"] = math.sqrt(mean_squared_error(lg_model.predict(X_train), y_train))
    lg_result["Test RMSE"] = math.sqrt(mean_squared_error(lg_model.predict(X_valid), y_valid))
    lg_result = pd.DataFrame(lg_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix,lg_result], axis = 0)

    # 릿지회귀
    ridge_model = Ridge(alpha = 0.1, solver="sag", random_state = 1004)
    ridge_model.fit(X_train, y_train)
    ridge_result = {}
    ridge_result["Model"] = "Ridge"
    ridge_result["Train score"] = ridge_model.score(X_train, y_train)
    ridge_result["Test score"] = ridge_model.score(X_valid, y_valid)
    ridge_result["Train RMSE"] = math.sqrt(mean_squared_error(ridge_model.predict(X_train), y_train))
    ridge_result["Test RMSE"] = math.sqrt(mean_squared_error(ridge_model.predict(X_valid), y_valid))
    ridge_result = pd.DataFrame(ridge_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix, ridge_result], axis = 0)

    # 라쏘회귀
    lasso_model = Lasso(alpha = 0.1, random_state = 1004)
    lasso_model.fit(X_train, y_train)
    lasso_result = {}
    lasso_result["Model"] = "Lasso"
    lasso_result["Train score"] = lasso_model.score(X_train, y_train)
    lasso_result["Test score"] = lasso_model.score(X_valid, y_valid)
    lasso_result["Train RMSE"] = math.sqrt(mean_squared_error(lasso_model.predict(X_train), y_train))
    lasso_result["Test RMSE"] = math.sqrt(mean_squared_error(lasso_model.predict(X_valid), y_valid))
    lasso_result = pd.DataFrame(lasso_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix, lasso_result], axis = 0)

    # 엘라스틱넷
    elastic_model = ElasticNet(alpha = 0.1, l1_ratio=0.5, random_state = 1004)
    elastic_model.fit(X_train, y_train)
    elastic_result = {}
    elastic_result["Model"] = "ElasticNet"
    elastic_result["Train score"] = elastic_model.score(X_train, y_train)
    elastic_result["Test score"] = elastic_model.score(X_valid, y_valid)
    elastic_result["Train RMSE"] = math.sqrt(mean_squared_error(elastic_model.predict(X_train), y_train))
    elastic_result["Test RMSE"] = math.sqrt(mean_squared_error(elastic_model.predict(X_valid), y_valid))
    elastic_result = pd.DataFrame(elastic_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix, elastic_result], axis = 0)

    # xgboost
    xg_model = XGBRegressor(n_estimators=200, learning_rate=0.15, gamma=0, subsample=0.35,
                            colsample_bytree=1, max_depth=3, random_state = 1004)
    xg_model.fit(X_train, y_train)
    xg_result = {}
    xg_result["Model"] = "XGBoostRegressor"
    xg_result["Train score"] = xg_model.score(X_train, y_train)
    xg_result["Test score"] = xg_model.score(X_valid, y_valid)
    xg_result["Train RMSE"] = math.sqrt(mean_squared_error(xg_model.predict(X_train), y_train))
    xg_result["Test RMSE"] = math.sqrt(mean_squared_error(xg_model.predict(X_valid), y_valid))
    xg_result = pd.DataFrame(xg_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix, xg_result], axis = 0)

    # 랜덤포레스트 회귀
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state = 1004)
    rf_model.fit(X_train, y_train)
    rf_result = {}
    rf_result["Model"] = "RandomForestRegressor"
    rf_result["Train score"] = rf_model.score(X_train, y_train)
    rf_result["Test score"] = rf_model.score(X_valid, y_valid)
    rf_result["Train RMSE"] = math.sqrt(mean_squared_error(rf_model.predict(X_train), y_train))
    rf_result["Test RMSE"] = math.sqrt(mean_squared_error(rf_model.predict(X_valid), y_valid))
    rf_result = pd.DataFrame(rf_result, index = ["Model"])
    result_matrix = pd.concat([result_matrix, rf_result], axis = 0)

    # result_matrix

    # Lasso or ElasticNet이 Good Model
    # 그럼 이제 예측에 사용할 데이터를 만들어보자~
    '''
    문제 1 : 2024년 타자 데이터가 온전하지 않다
    따라서, prediction 을 위한 input가 온전하지 않다.
    현재 구할 수 있는 2024년 데이터는 봄 훈련시 데이터 뿐 => 이를 input 데이터로 활용하기에는 좀 불완전하다.
    차선 해결책 1 : 타자 리스트는 2023년 타자 명단을 활용한다. + 2024년 데이터도 같이
    차선 해결책 2 : 예측값으로 활용할 feature values가 필요하다
    '''
    target_hitter = pd.concat([hitter_2023["PLAYER"], hitter_2024["PLAYER"]], axis = 0)
    target_hitter = target_hitter.unique()
    #target_hitter

    import warnings
    warnings.filterwarnings("ignore")
    # 방법 1 : hitter_total의 평균값을 이용해서 hitter_2023에 넣어서 활용 => 뭔가가 뭔가 문제가 있어서 57개밖에 안가져와진다...
    # 방법 2 : 2018년부터 2024년까지의 평균 데이터를 이용해 2024년 예측 Input으로 활용해보자
    # 2018년부터 2023년까지 싹다 불러옴
    hitter_2018 = pd.read_csv(".\hitter_2018.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2019 = pd.read_csv(".\hitter_2019.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2020 = pd.read_csv(".\hitter_2020.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2021 = pd.read_csv(".\hitter_2021.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2022 = pd.read_csv(".\hitter_2022.csv", encoding = "UTF-8", index_col = "RANK")
    hitter_2023 = pd.read_csv(".\hitter_2023.csv", encoding = "UTF-8", index_col = "RANK")
    # hitter_2024 = pd.read_csv(".\hitter_2024.csv", encoding = "UTF-8", index_col = "RANK") -> 24년은 데이터 수가 너무적어 패스

    # 위에서 했던 전처리 친구들을 빠르게 진행
    df_list = ["hitter_2018", "hitter_2019", "hitter_2020", "hitter_2021", "hitter_2022", "hitter_2023"] # 반복할 리스트 정의
    for i in df_list:
        if i[0:6] == "hitter" and i[-1] != "l" : # total 친구는 빼고 진행할거라 변수를 반복하도록 해줌
            if "Unnamed: 0" in globals()[i].columns :
                globals()[i] = globals()[i].drop("Unnamed: 0", axis = 1)
            globals()[i].columns = ["PLAYER", "POSITION", "TEAM", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS"]
            col_list = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS"]

            globals()[i] = globals()[i].drop(["POSITION", "TEAM"], axis = 1) # 얘넨 어차피 안쓸거라 제외

            globals()[i].fillna(0, inplace = True)

            globals()[i].loc[globals()[i]["CS"] == "--", "CS"] = 0 
            globals()[i].loc[globals()[i]["SB"] == "--", "SB"] = 0
            # 변수 정수형으로 변경
            for j in col_list :
                globals()[i][j] = globals()[i][j].astype("int64")

            # 변수 실수형으로 변경
            col_list = ["AVG", "OBP", "SLG", "OPS"]
            for j in col_list :
                globals()[i][j] = globals()[i][j].astype("float")
            
            # 교차항 생성
            col_names = ["G", "AB", "R", "H", "2B", "3B", "RBI", "BB", "SO", "SB", "CS", "AVG", "OBP", "SLG", "OPS"]
            temp_hitter_df = pd.DataFrame()
            for j in col_names :
                # temp = globals()[i].drop("PLAYER", axis = 1)
                # temp = temp.corr()[j] > 0.5
                n = 0
                for x in col_names :
                    if x != "HR" :
                        if (j != x) :
                            if j+"*"+x in temp_hitter_df.columns :
                                # print(j,"*",x, "가 있어서 스킵")
                                continue
                            elif x+"*"+j in temp_hitter_df.columns :
                                # print(x,"*",j,"가 있어서 스킵")
                                continue
                            else:
                                temp_hitter_df[x+"*"+j] = globals()[i][j] * globals()[i][x]
                                temp_hitter_df[j+"*"+x] = globals()[i][j] * globals()[i][x]
                        n += 1
            # print(temp_hitter_df.columns)
            globals()["transformed_"+i] = pd.DataFrame()
            col_names_1 = ["PLAYER","3B", "HR", "RBI", "SO", "SLG"]
            for x in col_names_1 :
                globals()["transformed_"+i] = pd.concat([globals()["transformed_"+i], globals()[i][x]], axis = 1)

            col_names_2 = ["RBI*G", "RBI*AB", "RBI*BB", "RBI*OPS", "BB*SO", "SO*RBI", "OBP*SO", "SLG*RBI", "SLG*OPS"]
            for x in col_names_2 :
                globals()["transformed_"+i] = pd.concat([globals()["transformed_"+i], temp_hitter_df[x]], axis = 1)

    # 최종적으로 예측을 수행할 데이터는 타자 목록에 대해서만 남겨서 데이터프레임을 줄여놓자
    for i in range(0, len(df_list)) :
        df_list[i] = "transformed_"+df_list[i]


    for i in df_list :
        temp_index = []
        target = globals()[i]
        print(i,"변환전 :",globals()[i].size)
        target["Target"] = ""
        for j in range(0,len(target["PLAYER"])) :
            for x in target_hitter :
                # print(target.iloc[j,0],"=",x,"?")
                if target.iloc[j,0] == x :
                    # print(target.iloc[j,0],"=",x,"!")
                    target.iloc[j,-1] = "Y"
                    # print(target.iloc[j,:])
                    # print(target.iloc[j,:])
        globals()[i] = target.drop(target[target["Target"] != "Y"].index, axis = 0)
        globals()[i] = globals()[i].drop("Target", axis = 1)
        print(i,"변환후 :",globals()[i].size)

    target_hitter = pd.DataFrame(target_hitter)
    target_hitter.columns = ["PLAYER"]
    # target_hitter

    # 값들을 하나의 데이터프레임으로 합쳐보자~
    diff_df = pd.DataFrame()
    for i in target_hitter["PLAYER"] :
        for j in df_list :
            target = globals()[j]
            for x in range(0, len(target["PLAYER"])):
                if target.iloc[x,0] == i :
                    temp_diff = target.iloc[x,:]
                    diff_df = pd.concat([diff_df, temp_diff], axis = 1)
    diff_df = diff_df.transpose()
    # diff_df
    # 값들의 평균을 구해보자
    mean_df = diff_df.groupby("PLAYER").mean()
    # mean_df
    # 앞에 학습시켰었던 (스케일러 적용전) 데이터 중 3B, HR, RBI, SO, RBI*G, RBI*AB, RBI*BB, BB*SO, SO*RBI는 정수형이다 반올림으로 맞춰주자
    col_names_1 = ["3B", "HR", "RBI", "SO", "SLG"]
    for i in col_names_1 :
        mean_df[i] = mean_df[i].astype("int")
    col_names_2 = ["RBI*G", "RBI*AB", "RBI*BB", "RBI*OPS", "BB*SO", "SO*RBI", "OBP*SO", "SLG*RBI", "SLG*OPS"]
    for i in col_names_2 :
        mean_df[i] = mean_df[i].astype("float")
    temp_list = ["3B", "HR", "RBI", "SO", "RBI*G", "RBI*AB", "RBI*BB", "BB*SO", "SO*RBI"]
    for i in temp_list :
        mean_df[i] = mean_df[i].round(0)
    # mean_df.head()

    skew_features = mean_df[features_index].apply(lambda x : skew(x))

    skew_features_top = skew_features[skew_features > 1]
    print(skew_features_top.sort_values(ascending = False))

    # 4개 친구들이 왜도가 높다 => 로그 스케일로 변환
    mean_df[skew_features_top.index] = np.log1p(mean_df[skew_features_top.index])

    # 다시 데이터의 왜곡 정도를 확인

    features_index = mean_df.dtypes[mean_df.dtypes != "object"].index

    skew_features = mean_df[features_index].apply(lambda x : skew(x))

    skew_features_top = skew_features[skew_features > 1]
    print(skew_features_top.sort_values(ascending = False))

    # 가장 성능이 좋았던 Lasso, ElasticNet 모델로 예측을 수행하자
    mean_df_drop = mean_df.drop("HR", axis = 1)
    final_prediction_scaled = elastic_model.predict(mean_df_drop)
    final_prediction_scaled = final_prediction_scaled.reshape(-1, 1)
    final_prediction_ela = final_prediction_scaled
    print(final_prediction_ela.mean())
    final_prediction_ela = pd.DataFrame(final_prediction_ela)
    final_prediction_ela.columns = ["HR"]
    final_prediction_ela = pd.concat([pd.DataFrame(mean_df.index), final_prediction_ela], axis = 1)
    final_prediction_ela.sort_values(by="HR", ascending = False)
    return final_prediction_ela