from bs4 import BeautifulSoup
import requests
from time import sleep
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import random

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns',None)


# 読み込んだデータの進捗を保存するためのリストを用意しておく
pre_results = []

#Engineer_Agent_Scraping
def EAS(url, pre_results=[]):

    e_list = pre_results

    if (len(e_list) > 0):
        progress = list(pd.DataFrame(e_list)['ページ'])
    else:
        progress = []

    #403 forbiddenのエラーを回避するために、requestsメソッドのヘッダーにユーザーエージェントの設定をする
    #利用規約でスクレイピング禁止を明言していないくせに...!
    headers = {
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
    }

    target_url = url.format(1)
    r = requests.get(target_url, headers = headers)
    sleep(random.randint(1,3))
    soup = BeautifulSoup(r.content, 'html.parser')
    end = soup.find('span', class_='project-list-header__num').text
    print("Scraping Start!")

    for page in tqdm(range(1, (int(end)//20+1)+1)):
        try:

            target_url = url.format(page)
            r = requests.get(target_url, headers = headers)
            sleep(random.randint(1,3))
            soup = BeautifulSoup(r.content, 'html.parser')
            contents = soup.find_all('a', class_='project-article-card__title')

            for page_num, content in enumerate(tqdm(contents)):
                if (int(str(page)+'{0:02d}'.format(page_num+1)) in progress):
                    continue

                #カラムは後から追加できるので、今はページのみ宣言
                e = {
                    "ページ":int(str(page)+'{0:02d}'.format(page_num+1))
                }
                
                print("page{}:".format(e["ページ"]),target_url)

                rc = requests.get(content.get('href'), headers = headers)
                sleep(random.randint(1,3))
                soupc = BeautifulSoup(rc.content, 'html.parser')

                e["タイトル"] = soupc.find('div', class_='project-article-card__title').text

                cont_names = soupc.find_all('div', class_='project-article-card__item-name')
                cont_values = soupc.find_all('div', class_='project-article-card__item-value')

                #スクレイピングした情報を格納(格納されなかったカラムはNaNになる)
                for cont_num, cont_name in enumerate(cont_names):
                    e[cont_name.text] = cont_values[cont_num].text

                e_list.append(e)

        except IndexError:
            continue
        except:
            pre_results = e_list
            return e_list
    
    pre_results = e_list
    return e_list


# 引数にpre_resultsを入れておけば、中断したところから素早く再開できる！
url = "https://tech-stock.com/projects/{}/?careerIds=&careerIds=&careerIds=&onlyRecruiting=true"
t_list = EAS(url, pre_results)
test = pd.DataFrame(t_list)
test

# pickleファイルに保存する
test.to_pickle('Engineer_Offers.pickle')
