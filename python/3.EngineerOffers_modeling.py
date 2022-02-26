import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


# 2.EngineerOffers_preprocessing.pyを読み込み、データの前処理を完了させる。
# データ読み込み
df = pd.read_pickle('Engineer_Agent_Bigdata.pickle')
df_pre = df.copy()
df.head(10)

# データ前処理
engineer_agent_preprocessing(df)
df.head(10)


# 重回帰分析
# データを学習用とテスト用に分割する
df_x = df.drop(['月額報酬'], axis=1)
df_y = df['月額報酬']
df_l_x, df_t_x, df_l_y, df_t_y = train_test_split(df_x, df_y, shuffle=True, test_size=0.2, random_state=71)

model_lr = LinearRegression()
model_lr.fit(df_l_x, df_l_y)

df_pred = pd.DataFrame(model_lr.predict(df_t_x), columns=['pred']).reset_index(drop=True)
df_true = pd.DataFrame(df_t_y).reset_index(drop=True).set_axis(['true'], axis=1)

print('MAE:', mean_absolute_error(df_pred, df_true))
print('RMSE', np.sqrt(mean_squared_error(df_pred, df_true)))
display(pd.concat([df_pred, df_true], axis=1))

# 残差プロットを書いてみる
plt.scatter(df_pred.pred, df_pred.pred - df_true.true, color = 'blue')
plt.hlines(y = 0, xmin = 20, xmax = 150, color = 'red')
plt.title('Residual Plot')
plt.xlabel('pred')
plt.ylabel('residuals')
plt.grid()
plt.show()


# ランダムフォレスト
# データを学習用とテスト用に分割する
df_x = df.drop(['月額報酬'], axis=1)
df_y = df['月額報酬']
df_l_x, df_t_x, df_l_y, df_t_y = train_test_split(df_x, df_y, shuffle=True, test_size=0.2, random_state=71)

model_rf = RandomForestClassifier(max_depth=7)
model_rf.fit(df_l_x, df_l_y.astype(int))
df_pred = pd.DataFrame(model_rf.predict(df_t_x), columns=['pred']).reset_index(drop=True)
df_true = pd.DataFrame(df_t_y).reset_index(drop=True).set_axis(['true'], axis=1)

print('MAE:', mean_absolute_error(df_pred, df_true))
print('RMSE', np.sqrt(mean_squared_error(df_pred, df_true)))
display(pd.concat([df_pred, df_true], axis=1))

# 残差プロットを書いてみる
plt.scatter(df_pred.pred, df_pred.pred - df_true.true, color = 'blue')
plt.hlines(y = 0, xmin = 20, xmax = 150, color = 'red')
plt.title('Residual Plot')
plt.xlabel('pred')
plt.ylabel('residuals')
plt.grid()
plt.show()

# 特徴量の重要度を見る
pd.DataFrame(model_rf.feature_importances_, index=df_x.columns, columns=['importance']).sort_values('importance', ascending=False)


# XGBOOST
df_x = df.drop(['月額報酬'], axis=1)
df_y = df['月額報酬']

# データを学習用とテスト用に分割する
tr_x, test_x, tr_y, test_y = train_test_split(df_x, df_y, shuffle=True, test_size=0.2, random_state=71)
# 学習用データを学習用と検証用に分割する
tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, shuffle=True, test_size=0.2, random_state=71)
display(tr_x.head(10))
display(tr_y.head(10))

#データ型をXGBOOST用に適合させる
dtrain = xgb.DMatrix(tr_x,label=tr_y)
dvalid = xgb.DMatrix(va_x,label=va_y)

#XGBOOSTのモデルを作成
num_round = 1000
early_stopping_rounds=50
params = {'objective':'reg:squarederror',
          'silent':1,
          'random_state':71
}
watchlist = [(dtrain,'train'),(dvalid,'eval')]
model_xgb = xgb.train(params,dtrain,num_round,early_stopping_rounds=early_stopping_rounds,evals=watchlist)

#テストデータとその予測結果を表示
test_pred = model_xgb.predict(xgb.DMatrix(test_x))
df_true = pd.DataFrame(list(test_y),columns=['月額報酬'])
df_pred = pd.DataFrame(test_pred,columns=['予測値'])
display(pd.concat([df_true,df_pred],axis=1))

#モデルの性能を表示
print('MAE:',mean_absolute_error(test_y,test_pred))
print('MSE:',mean_squared_error(test_y,test_pred))
print('RMSE:',np.sqrt(mean_squared_error(test_y,test_pred)))


# アプリ化(自分の好きなデータを入れて予測してみる)
pred_data = pd.DataFrame({
    '勤務地':LE_office.transform(['確認中']),
    '勤務期間':12, # 1年間 = 12
    '稼働率':100, # パーセント
    '業種':LE_type.transform(['非公開']),
    'リモート':LE_remote.transform(['なし（常駐）']),
    'ポジション':0, # メンバー:0、サブリーダー:1、リーダー:2
    '募集人数':1, 
    '服装':1, # スーツネクタイ着用:0, ビジネスカジュアル:1, 私服可:2
    '契約形態':0, # 業務委託:0, 派遣:1, 契約社員:2
    '延長':0,
    
    # 職業名
    'その他オフィスワーク':0, 
    'システムエンジニア・プログラマー':0, 
    'インフラ・ネットワークエンジニア':0, 
    'ネットワークエンジニア':0,
    'コンサル・PM・PMO':0,
    'テスター・デバッカー':0, 
    'フロントエンドエンジニア':0,
    'バックエンドエンジニア':0,
    'デザイナー・クリエイター':0,
    'Java系エンジニア':0,
    '業務系エンジニア':0,
    'サーバエンジニア':0,
    '運用/監視担当':0,
    '企画・マーケティング':0,
    'データサイエンティスト':0,
    'DBA(データベース)管理者':0,
    'ヘルプデスク':0,
    'スマホアプリ開発':0,
    'ソーシャル系エンジニア':0,
    '制御・組み込み系エンジニア':0,
    'ゲームプログラマー/ゲームクリエイター':0,
    'アナリスト':0,
    'カスタマーサポート':0,
    'Webディレクター':0,
    'SAP系(ABAP・BASIS)エンジニア':0,
    'Windows系エンジニア':0,
    'LAMP系エンジニア':0,
    
    # 英語スキル
    'Reading':1,
    'Listening':1,
    'Speaking':0,
    'Writing':0
}, index=['pred'])
display(pred_data)

pred = model_xgb.predict(xgb.DMatrix(pred_data))
display(pd.DataFrame(pred,columns=['月額報酬']))


# LightGBM
# データを学習用とテスト用に分割する
df_x = df.drop(['月額報酬'], axis=1)
df_y = df['月額報酬']

# データを学習用とテスト用に分割する
tr_x, test_x, tr_y, test_y = train_test_split(df_x, df_y, shuffle=True, test_size=0.2, random_state=71)
# 学習用データを学習用と検証用に分割する
tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, shuffle=True, test_size=0.2, random_state=71)
display(tr_x.head(10))
display(tr_y.head(10))

# カテゴリ変数をそのまま使う
# データ型をLIGHTGBM用に適合させる。
ltrain = lgb.Dataset(tr_x,tr_y)
lvalid = lgb.Dataset(va_x,va_y,reference=ltrain)

# パラメータ準備
params = {'objective':'regression','metrics':'rmse','silent':1,'random_state':71,
          'max_depth':7,
          'min_child_weight':2,
          'gamma':0.2,
          'colsample_bytree':1.0,
          'colsample_bylevel':0.3,
          'subsample':0.07,
          'alpha':1,
          'eta':0.1, 
          #'lambda':1
         }
num_round = 1000
early_stopping_rounds = 50

# LIGHTGBMモデルに機械学習
categorical_feature = ['勤務地', '業種', 'リモート', 'ポジション', '服装', '契約形態']
model_lgb = lgb.train(params,ltrain,num_boost_round=num_round,
                  early_stopping_rounds=early_stopping_rounds,
                  valid_names=['train','valid'],valid_sets=[ltrain,lvalid],
                 categorical_feature=categorical_feature)
test_pred = model_lgb.predict(test_x)

# モデルの性能を表示させる。
print('MAE:',mean_absolute_error(test_y,test_pred))
print('MSE:',mean_squared_error(test_y,test_pred))
print('RMSE:',np.sqrt(mean_squared_error(test_y,test_pred)))

# テストデータとその予測結果を表示させる。
test_y_df = pd.DataFrame(test_y).reset_index(drop=True)
test_pred_df = pd.DataFrame(test_pred,columns=['prediction'])
display(pd.concat([test_y_df,test_pred_df],axis=1))


# アプリ化(自分の好きなデータを入れて予測してみる)
pred_data = pd.DataFrame({
    '勤務地':LE_office.transform(['確認中']),
    '勤務期間':12, # 1年間 = 12
    '稼働率':100, # パーセント
    '業種':LE_type.transform(['非公開']),
    'リモート':LE_remote.transform(['なし（常駐）']),
    'ポジション':0, # メンバー:0、サブリーダー:1、リーダー:2
    '募集人数':1,
    '服装':0, # スーツネクタイ着用:0, ビジネスカジュアル:1, 私服可:2
    '契約形態':0, # 業務委託:0, 派遣:1, 契約社員:2
    '延長':0,
    
    # 職業名
    'その他オフィスワーク':0,
    'システムエンジニア・プログラマー':0, 
    'インフラ・ネットワークエンジニア':0, 
    'ネットワークエンジニア':0,
    'コンサル・PM・PMO':0,
    'テスター・デバッカー':0, 
    'フロントエンドエンジニア':0,
    'バックエンドエンジニア':0,
    'デザイナー・クリエイター':0,
    'Java系エンジニア':0,
    '業務系エンジニア':0,
    'サーバエンジニア':0,
    '運用/監視担当':0,
    '企画・マーケティング':0,
    'データサイエンティスト':1,
    'DBA(データベース)管理者':0,
    'ヘルプデスク':0,
    'スマホアプリ開発':0,
    'ソーシャル系エンジニア':0,
    '制御・組み込み系エンジニア':0,
    'ゲームプログラマー/ゲームクリエイター':0,
    'アナリスト':0,
    'カスタマーサポート':0,
    'Webディレクター':0,
    'SAP系(ABAP・BASIS)エンジニア':0,
    'Windows系エンジニア':0,
    'LAMP系エンジニア':0,
    
    # 英語スキル
    'Reading':1,
    'Listening':1,
    'Speaking':0,
    'Writing':0
}, index=['pred'])
display(pred_data)

pred = model_lgb.predict(pred_data)
display(pd.DataFrame(pred,columns=['月額報酬'],index=['予測値(万円)']))
