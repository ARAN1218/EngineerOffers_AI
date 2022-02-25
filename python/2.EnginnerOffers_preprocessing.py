def import_test():
    print("All Functions Imported!")
    
    
def page_preprocessing(df_page):
    del df_page['ページ']
    
    print('page preprocessed!')


def title_preprocessing(df_title):
    del df_title['タイトル']
    
    print("title preprocessed!")
    

def reward_preprocessing(df_reward):
    #00~00の形だけ残す
    df_reward['月額報酬'] = df_reward['月額報酬'].str.translate(str.maketrans({'\n':'', '万':'', '円':'', '(':'', ')':'', '税':'', '別':''}))

    #足して2で割る
    df_reward['月額報酬'] = df_reward['月額報酬'].map(lambda x : (int(x.split('～')[0])+int(x.split('～')[1]))/2 if ('～' in x) else int(x))

    print('reward preprocessed!')


def office_preprocessing(df_office):
    #タグ等を除去する
    df_office['勤務地'] = df_office['勤務地'].str.translate(str.maketrans({'\n':''
                                                                     , ' ':''
                                                                     , '\u3000':''
                                                                     ,'・':''
                                                                     #,'':''
                                                                     , '│':'/'
                                                                     , '／':'/'
                                                                     ,'+':'/'
                                                                     ,'＋':'/'
                                                                     ,'※':'/'
                                                                     ,'(':'/'
                                                                     ,'（':'/'
                                                                    }))
    
    #「/」で区切って、左の要素を取得する。
    df_office['勤務地'] = df_office['勤務地'].map(lambda x : x.split('/')[0] if ('/' in x) else x)

    #地名からリモートを消去し、リモートを表す言葉を「リモート」で統一したい
    eraces = [
        'フルリモート'
        ,'現状リモート勤務'
        ,'、リモート勤務'
        ,'基本リモート'
        ,'原則リモート'
        ,'リモート勤務'
        ,'一部リモート'
        ,'、リモート'
        ,'テレワーク'
        ,'orリモート'
        ,'リモート、'
        ,'リモート'
    ]

    for erace in eraces:
        df_office['勤務地'] = df_office['勤務地'].str.replace(erace, '')
    df_office['勤務地'][df_office['勤務地']==''] = 'リモート'
    
    #ラベルエンコーディング
    from sklearn.preprocessing import LabelEncoder

    global LE_office
    LE_office = LabelEncoder()
    LE_office.fit(df_office['勤務地'])
    df_office['勤務地'] = LE_office.transform(df_office['勤務地'])
    
    print('office preprocessed!')
    print(LE_office.classes_)


def span_preprocessing(df_span):
    #延長カラムを作っていく
    #「延長」「更新」という単語が含まれている行は1、それ以外は0にする
    df_span['延長'] = df_span['勤務期間'].map(lambda x : 1 if (('延長' in x) or ('更新' in x)) else 0)
    
    # 即日を2021年7月長期を2024年7月(=3年)と定義し、変換する
    df_span['勤務期間'] = df_span['勤務期間'].str.replace('即日', '2021年7月')
    df_span['勤務期間'] = df_span['勤務期間'].str.replace('長期', '2024年7月')
    
    #不要な文字を削除する。
    df_span['勤務期間'] = df_span['勤務期間'].str.translate(str.maketrans({'\n':''
                                                                   , ' ':''
                                                                   ,'月':'時'
                                                                   ,'中':''
                                                                   , '(':'|'
                                                                   ,'~':'～'
                                                                  }))
    
    #「|」の左側だけ切り取る
    df_span['勤務期間'] = df_span['勤務期間'].map(lambda x : x.split('|')[0] if ('|' in x) else x)
    
    #「～」で分ける。
    df_span['1'] = df_span['勤務期間'].map(lambda x : x.split('～')[0] if ('～' in x) else x)
    df_span['2'] = df_span['勤務期間'].map(lambda x : x.split('～')[1] if ('～' in x) else 0)
    
    #「時」の左側だけ切り取る
    df_span['1'] = df_span['1'].map(lambda x : x.split('時')[0] if ('時' in str(x)) else x)
    df_span['2'] = df_span['2'].map(lambda x : x.split('時')[0] if ('時' in str(x)) else x)
    
    # 不明な文字列は2022年9月に変換
    # 数値はstrにして2021年と足す
    df_span['1'][df_span['1'] == '8'] = '2021年8'
    df_span['1'][df_span['1'] == '2021/9/1'] = '2021年9'
    df_span['2'][df_span['2'] == '2021/9/30'] = '2021年9'
    df_span['2'] = df_span['2'].apply(lambda x : '2021年'+str(x) if x in ['7', '8', '9', '10', '11', '12'] else x)
    df_span['2'] = df_span['2'].apply(lambda x : '2022年7' if x in ['', '終了', '1年想定', '確認', 0] else x)
    
    # datetime型にして月単位で差をとる
    from datetime import datetime
    for i in range(len(df_span)):
        df_span['勤務期間'][i] = ((datetime.strptime(df_span['2'][i], '%Y年%m') - datetime.strptime(df_span['1'][i], '%Y年%m')).days + 1) // 30
        
    df_span['勤務期間'] = df_span['勤務期間'].astype('int')
    
    del df_span['1']
    del df_span['2']

    print("span preprocessed!")


def per_preprocessing(df_per):
    #余計な文字列を取り除く
    df_per['稼働率'] = df_per['稼働率'].str.translate(str.maketrans({'\n':'', ' ':'', '%':''}))

    #「～」で分割する
    df_per['1'] = df_per['稼働率'].map(lambda x : int(x.split('～')[0]) if ('～' in x) else int(x))
    df_per['2'] = df_per['稼働率'].map(lambda x : int(x.split('～')[1]) if ('～' in x) else int(x))

    #1と2の平均値を特徴量とする
    df_per['稼働率'] = (df_per['1'] + df_per['2']) / 2
    
    del df_per['1']
    del df_per['2']
    
    print('per preprocessed!')


def task_preprocessing(df_task):
    del df_task['業務内容']
    
    print("task preprocessed!")


def skill_preprocessing(df_skill):
    del df_skill['スキル']
    
    print("skill preprocessed!")


def type_preprocessing(df_type):
    df_type['業種'] = df_type['業種'].str.translate(str.maketrans({'\n':'', ' ':''}))
    
    #ラベルエンコーディング
    from sklearn.preprocessing import LabelEncoder
    
    global LE_type
    LE_type = LabelEncoder()
    LE_type.fit(df_type['業種'])
    df_type['業種'] = LE_type.transform(df_type['業種'])
    
    print('type preprocessed!')
    print(LE_type.classes_)


def remote_preprocessing(df_remote):
    df_remote['リモート'] = df_remote['リモート'].str.translate(str.maketrans({'\n':'', ' ':''}))
    df_remote['リモート'][df_remote['リモート'] == ''] = 'なし（常駐）'
    
    #ラベルエンコーディング
    from sklearn.preprocessing import LabelEncoder
    
    global LE_remote
    LE_remote = LabelEncoder()
    LE_remote.fit(df_remote['リモート'])
    df_remote['リモート'] = LE_remote.transform(df_remote['リモート'])
    df_remote['リモート'].unique()
    
    print("remote preprocessed!")
    print(LE_remote.classes_)


def job_preprocessing(df_job):
    df_job['募集職種'] = df_job['募集職種'].str.translate(str.maketrans({'\n':'', ' ':''}))
    
    # ダミー変数化
    jobs = [
        'その他オフィスワーク',
        'システムエンジニア・プログラマー',
        'インフラ・ネットワークエンジニア',
        'ネットワークエンジニア',
        'コンサル・PM・PMO',
        'テスター・デバッカー',
        'フロントエンドエンジニア',
        'バックエンドエンジニア',
        'デザイナー・クリエイター',
        'Java系エンジニア',
        '業務系エンジニア',
        'サーバエンジニア',
        '運用/監視担当',
        '企画・マーケティング',
        'データサイエンティスト',
        'DBA(データベース)管理者',
        'ヘルプデスク',
        'スマホアプリ開発',
        'ソーシャル系エンジニア',
        '制御・組み込み系エンジニア',
        'ゲームプログラマー/ゲームクリエイター',
        'アナリスト',
        'カスタマーサポート',
        'Webディレクター',
        'SAP系(ABAP・BASIS)エンジニア',
        'Windows系エンジニア',
        'LAMP系エンジニア'
    ]

    for job in jobs:
        df_job[job] = df_job['募集職種'].map(lambda x : 1 if job in x else 0)
        df_job['募集職種'] = df_job['募集職種'].str.replace(job, '')
        
    del df_job['募集職種']
        
    print("job preprocessed!")


def position_preprocessing(df_position):
    df_position['ポジション'] = df_position['ポジション'].str.translate(str.maketrans({'\n':'', ' ':''}))
    
    # メンバー:0、サブリーダー:1、リーダー:2
    df_position['ポジション'] = df_position['ポジション'].str.replace('サブリーダー', 'サブリーダ')
    df_position['ポジション'] = df_position['ポジション'].apply(lambda x : 2 if ('リーダー' in x) else 1 if 'サブリーダ' in x else 0)
    
    print("position preprocessed!")


def people_preprocessing(df_people):
    df_people['募集人数'] = df_people['募集人数'].str.translate(str.maketrans({'\n':'', ' ':'', '名':''}))
    
    # NaNは1に置き換える
    df_people['募集人数'] = df_people['募集人数'].fillna(1)
    df_people['募集人数'] = df_people['募集人数'].astype(int)
    
    print("people preprocessed!")


def time_preprocessing(df_time):
    del df_time['勤務時間']
    
    print("time preprocessed!")


def fashion_preprocessing(df_fashion):
    df_fashion.服装 = df_fashion.服装.str.translate(str.maketrans({'\n':'', ' ':''}))
    
    # NaNはビジネスカジュアルとする
    df_fashion.服装 = df_fashion.服装.fillna('ビジネスカジュアル')
    
    # スーツネクタイ着用=0, ビジネスカジュアル=1, 私服可=2
    df_fashion.服装[df_fashion['服装'] == 'スーツネクタイ着用'] = 0
    df_fashion.服装[df_fashion['服装'] == 'ビジネスカジュアル'] = 1
    df_fashion.服装[df_fashion['服装'] == '私服可'] = 2
    
    df_fashion.服装 = df_fashion.服装.astype('int')
    
    print("fashion preprocessed!")


def contract_preprocessing(df_contract):
    df_contract.契約形態 = df_contract.契約形態.str.translate(str.maketrans({'\n':'', ' ':''}))
    
    # 紹介予定派遣を派遣に組み込む
    df_contract.契約形態[df_contract['契約形態'] == '紹介予定派遣'] = '派遣'
    
    # 業務委託=0, 派遣=1, 契約社員=2
    df_contract.契約形態[df_contract['契約形態'] == '業務委託'] = 0
    df_contract.契約形態[df_contract['契約形態'] == '派遣'] = 1
    df_contract.契約形態[df_contract['契約形態'] == '契約社員'] = 2
    
    df_contract.契約形態 = df_contract.契約形態.astype('int')
    
    print("contract preprocessed!")


def sub_preprocessing(df_sub):
    del df_sub['備考']
    
    print("sub preprocessed!")


def env_preprocessing(df_env):
    del df_env['開発環境']
    
    print("env preprocessed!")


def english_preprocessing(df_english):
    df_english.英語力 = df_english.英語力.str.translate(str.maketrans({'\n':'', ' ':''}))
    
    # イテレートするためにNaNを取り除かないとならないらしい
    df_english.英語力 = df_english.英語力.fillna('')

    for eng in ['Reading', 'Listening', 'Speaking', 'Writing']:
        df_english[eng] = df_english.英語力.map(lambda x : 1 if eng in x else 0)
        df_english.英語力 = df_english.英語力.str.replace(eng, '')

    del df_english['英語力']
    
    print("english preprocessed!")


def engineer_agent_preprocessing(df):
    import warnings
    warnings.simplefilter('ignore')
    page_preprocessing(df)
    title_preprocessing(df)
    reward_preprocessing(df)
    office_preprocessing(df)
    span_preprocessing(df)
    per_preprocessing(df)
    task_preprocessing(df)
    skill_preprocessing(df)
    type_preprocessing(df)
    remote_preprocessing(df)
    job_preprocessing(df)
    position_preprocessing(df)
    people_preprocessing(df)
    time_preprocessing(df)
    fashion_preprocessing(df)
    contract_preprocessing(df)
    sub_preprocessing(df)
    env_preprocessing(df)
    english_preprocessing(df)
    print('all preprocessing completed!!')
