#!/usr/bin/env python
# coding: utf-8

"""
        THIS PROGRAM IS READY TO BE RUN 
"""
"""
           Import Libraries
"""
import json
import numpy as np
import datetime
import requests
import os
import sys
import pandleau
from pandleau import *
from pandas.io.json import json_normalize
sys.path.append(os.path.abspath("C:\Tableau\hyper\tableausdk"))
sys.path.append(os.path.abspath("C:\Tableau\hyper\tableausdk\bin"))
sys.path.append(os.path.abspath("C:\Tableau\hyper\tableausdk\lib"))
import pandleau as pnd
sys.path.append(os.path.abspath("R:\_Department\Analytics\pythonlib"))
from hbbutils import *
from sql_utils import *
import tableau_utils

"""
     Create Temp Directroy if does not exist
"""

def create_temp_directory():
    directroy = 'C:\Temp'
    if not os.path.exists(directroy):
        os.makedirs(directroy)
    else:
        print('Temp Exists')
    

"""
    Function for Reading the JSON Object from API - Fixed Column JSON Object
"""
def read_json_obj_hc(json_obj):
    json_data = json_obj
    string_to_write = ""
    for i in range(len(json_data["data"])):
        string_to_write_tmp = json_data["data"][i]["id"] + "," + json_data["data"][i]["productId"] + "," + json_data["data"][i]["retailer"] + "," + json_data["data"][i]["reviewId"] + "," + str(json_data["data"][i]["starRating"]) + "," + json_data["data"][i]["title"] + "," + json_data["data"][i]["review"] + "," + json_data["data"][i]["date"]
        string_to_write_tmp = string_to_write_tmp.replace("\n"," ")
        string_to_write = string_to_write + string_to_write_tmp + "\n"

    return string_to_write


"""
    Function for Reading the JSON Object from API - Fixed Column JSON Object
"""
def read_json_obj(json_obj,last_review_date,file_name='C:\Temp\Product_Reviews_1126.csv'):
    json_data = json_obj
    #print (len(json_data["data"]))
    i=0
    str_to_write = ""
    for value in json_data["data"]:
            if i%3000 == 0:
                write_to_file(file_name,str_to_write)
                str_to_write=""
            else:
              dic=dict(value)  
              str_to_write_tmp=""
              str_to_write_tmp=last_review_date+"}"
              for key, value in dic.items():
                  value_str = str(value).replace("}"," ")
                  str_to_write_tmp = str_to_write_tmp + value_str + "}"

              str_to_write_tmp = str_to_write_tmp.replace("\n"," ") 
              str_to_write_tmp = str_to_write_tmp[:-1] + "\n"
              str_to_write = str_to_write + str_to_write_tmp

            i=i+1 
    
    if len(str_to_write) > 1:
        write_to_file(file_name,str_to_write)
        str_to_write=""
     
        
def write_to_file(file_name,string):
    with open(file_name, "a+", encoding="utf-8") as write_file:
        write_file.write(string)
        
def delete_file(file_name):
    try:
        os.remove(file_name)    
    except:
        print("Continue")
        pass

def api_reviews(api_header):
    headers = api_header
        
def urls(api_type):
    #print ("Api Type = " + api_type)
    if api_type == "Reviews":
        url="https://hamiltonbeach.reviewcamp.com/api/reviews"
    elif api_type == "Products":
        url = 'https://hamiltonbeach.reviewcamp.com/api/products'
    
    #print (url)
    return url

"""
reviewsAfter          --> Filter reviews equal to and after a specific date (format: YYYY-MM-DD)
reviewsBefore          --> Filter reviews equal to and after a specific date (format: YYYY-MM-DD)

"""
def query_Arguments(api_type,reviewsAfter='2001-01-01',reviewsBefore='',page=0):
    
    if api_type == "Reviews":
        queryString = {"per_page":700,"reviewsAfter":reviewsAfter,"reviewsBefore":reviewsBefore,'page': page}
    elif api_type == "Products":
        queryString = {"per_page":1000,'page': page}
    else:
        queryString = "Hello"
        
    print(queryString,   "     ", datetime.datetime.now())
    return queryString

def header_information():
    _headers = {
    'Authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjU3ODdkOTgxNGExYmNiMzlmMWIyMWUzZDlhMDI5MmU2OGEyY2M5ZGYwN2Y0YTY3MGYxNDA1ZmY2YmExMzIzNmNjMzgzZmQ2NTIxODFiMmFiIn0.eyJhdWQiOiIxIiwianRpIjoiNTc4N2Q5ODE0YTFiY2IzOWYxYjIxZTNkOWEwMjkyZTY4YTJjYzlkZjA3ZjRhNjcwZjE0MDVmZjZiYTEzMjM2Y2MzODNmZDY1MjE4MWIyYWIiLCJpYXQiOjE1NTkzNTc2NTMsIm5iZiI6MTU1OTM1NzY1MywiZXhwIjoxNTkwOTgwMDUzLCJzdWIiOiI4NCIsInNjb3BlcyI6W119.FoJxxGtJanvyKi0K5miLv2NyrNLx56Kif9iIcsIWOw3fQQ3-lLJYNm2JU7jmEzW0o7NXIyaUs-Q7i9h2Xj4w_rup3VUBRQJ5X9zMELUd0XFdyI6l2h0dMbuey45hkECGjlHQe33GyrL64alsQkGKkxNzuiK2wi37IghGr1EUgztO1-kw1-OB-G5Tn20374NBirHTFxgQ3Gxkj_IKHSSy1jaDZh5C2ZlI1ljNRFRd6ErOdjzSI8J1a1ASAFWz1_92C4Buulp556iOVcWqGdnUgkZ2gzIkpcvg0uFxwnVQ_xtrypdYXce1MeG1GP9Yq-XnieFU6NT-semW4HgMqD5bFFUwV-7-Ek2nbwGHjvMoC_5ouMywa8dW4AhK6omjXAceMyZXQYAB1FycRRiBADn8V6ENmU4UF9bpd52efN7yjnI1rWMHiSgyc14W0DYobbEzaT2imP7l1OMy2rKCISezBF9fGqocjm6hbHoPxORLH3wbQ_OQEHvwAqCMakppD53IXveIPyv9u7z6byjpPpH1tfKmqMHm19sv8qze7yO2egqS126aOxGzVn4e24dD55KJRMQRRA2EYU6rKwslHDcosuWC6D-6SUytxW_mZiOWOYcQHoR8jdOzPPbJ7GoQIHQdQprAUyyOHfVKXFAdgX2zeObHV1jbxRXSuOr-L0a6umY",
    'Content-Type': "application/json",
    }
    return _headers
    
def get_last_review_date(table_Name, schema_name):
    cxn = sql_connect_py('HB-SQL-DEV01','HBBCPOPR')
    cursor = cxn.cursor()
    cursor.execute("SELECT DATEADD(day,-3,MAX(Review_Date)) Review_Date FROM HBBCPOPR."+schema_name+"."+table_Name+" ;")
    results = cursor.fetchall() 
    i=0
    for row in results:
        review_date = row.Review_Date
        
    cxn.commit()
    cxn.close()
    
    return review_date    

def del_sql(table_Name, schema_name,last_review_date):
    cxn = sql_connect_py('HB-SQL-DEV01','HBBCPOPR')
    print ("Delete file  - last Review date",last_review_date)
    cxn.execute("Delete from HBBCPOPR."+schema_name+"."+table_Name+" Where Review_Date >='"+last_review_date+"';")
    cxn.commit()
    cxn.close()

def execute_sql(sp_name, schema_name):
    cxn = sql_connect_py('HB-SQL-DEV01','HBBCPOPR')
    str_exec = "execute HBBCPOPR."+schema_name+"."+sp_name+" ;"
    cxn.execute(str_exec)
    cxn.commit()
    cxn.close()
    
def truncate_sql(table_Name, schema_name):
    cxn = sql_connect_py('HB-SQL-DEV01','HBBCPOPR')
    cxn.execute("Truncate table HBBCPOPR."+schema_name+"."+table_Name+" ;")
    cxn.commit()
    cxn.close()
    
###   Function to load data into SQL Server
def load_to_sql(table_Name, schema_name, file_name):
    bcp_str =  "bcp " + schema_name + "." + table_Name + " in " + file_name + " -e " +file_name + ".err -T -c -t { -S HB-SQL-DEV01.DataCenter.hambeach.com -d HBBCPOPR"
    print(bcp_str)
    os.system(bcp_str)
    #os.remove(file_name)

def products():
    headers = header_information()
    truncate_sql("PRODUCTS","RCAMP")
    queryString = query_Arguments("Products")
    # Actual API CALL to reviewcamp website for Products
    response = requests.request("GET", urls("Products"), headers=headers, params=queryString)
    json_data = response.json() 
    file_name = 'C:\Temp\Products_1.csv'
    last_review_date = '2010-01-01'
    read_json_obj(json_data,last_review_date,file_name)
    
def review_extract_api(file_name,reviewsAfter='',reviewsBefore=''):
    
    print (" From Review Extract API ", reviewsAfter, '       ',reviewsBefore)
    headers = header_information()

    queryString = query_Arguments("Reviews",reviewsAfter,reviewsBefore)


    # Actual API CALL to reviewcamp website for reviews
    response = requests.request("GET", urls("Reviews"), headers=headers, params=queryString)
    json_data = response.json() 
    process_review_data(json_data,file_name)
    
    z=json_normalize(json_data['meta']['pagination'])
    number_pages=z['total_pages'][0]
    total_reivews = z['total'][0]
    print('Total Pages = ', number_pages, ' Total Reviews = ',total_reivews)
    
    for page in range (2,number_pages+1):
        queryString = query_Arguments("Reviews",reviewsAfter,reviewsBefore,page=page)
        response = requests.request("GET", urls("Reviews"), headers=headers, params=queryString)
        json_data = response.json() 
        process_review_data(json_data,file_name)

def process_review_data(dataf,file_name):
    x=json_normalize(dataf['data'])
    x.fillna(0,inplace=True)
    x=x[['id', 'productId','retailer', 'title','reviewId','starRating','review','format','date']]
    x['LDDate']=datetime.datetime.now().strftime('%Y-%m-%d')
    cols = x.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    x = x[cols]
    trimNewline = lambda x: x.replace("{","") if type(x) is str else x
    x=x.applymap(trimNewline)
    
    trimNewline = lambda x: x.replace("}","") if type(x) is str else x
    x=x.applymap(trimNewline)
    
    x.to_csv(file_name, sep='{',mode="a",index=False,header=False)
    

def product_extract_api(file_name):
    headers = header_information()

    queryString = query_Arguments("Products")
    response = requests.request("GET", urls("Products"), headers=headers, params=queryString)
    json_data = response.json() 
    process_product_data(json_data,file_name)
    
    z=json_normalize(json_data['meta']['pagination'])
    number_pages=z['total_pages'][0]
    total_reivews = z['total'][0]
    print('Total Pages = ', number_pages, ' Total Products = ',total_reivews)
    
    for page in range (2,number_pages+1):
        queryString = query_Arguments("Products",page=page)
        response = requests.request("GET", urls("Products"), headers=headers, params=queryString)
        json_data = response.json() 
        process_product_data(json_data,file_name)
        
def process_product_data(dataf,file_name):
    x=json_normalize(dataf['data'])
    x.fillna(0,inplace=True)
    x['LDDate']=datetime.datetime.now().strftime('%Y-%m-%d')
    cols = x.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    x = x[cols]
    #print("1 = ",cols)
    x=x[['LDDate','id','retailer','retailerId','name','sku','category', 'subcategory','brand','manufacturer','releasedAt','image',\
         'totalReviews', 'totalStars', 'average','retailerAverage','r3', 'r6','r12', 'fiveStars', 'fourStars','threeStars',\
         'twoStars', 'oneStars','user.data.username']]
    #print("2 = ",x.columns.tolist())
    trimNewline = lambda x: x.replace("{","") if type(x) is str else x
    x=x.applymap(trimNewline)
    
    trimNewline = lambda x: x.replace("}","") if type(x) is str else x
    x=x.applymap(trimNewline)
    
    x.to_csv(file_name, sep='{',mode="a",index=False,header=False)
    
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return (next_month - datetime.timedelta(days=next_month.day)).strftime("%Y-%m-%d")

def monthlist_fast(dates):
    start, end = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        m_sub=[]
        y, m = divmod(tot_m, 12)
        m_sub.append(datetime.datetime(y, m+1, 1).strftime("%Y-%m-%d"))
        m_sub.append(last_day_of_month(datetime.datetime(y, m+1, 1)))
        
        mlist.append(m_sub)
    return mlist

def complete_refresh(reviewsafter,reviewsbefore=''):
    truncate_sql("REVIEWS_FULL_REFRESH","RCAMP")
    if reviewsbefore=="":
        reviewsbefore=datetime.datetime.now().strftime('%Y-%m-%d')
    dates = [reviewsafter, reviewsbefore]
    print(dates)
    monthlist_fast(dates)

    for i in monthlist_fast(dates):
        file_name = "C:\\temp\\Reviews\\Reviews_"+i[0]+".xlsx"
        print(file_name)
        delete_file(file_name)
        print('review_extract_api(reviewsAfter='+i[0]+"  "+i[1])
        review_extract_api(file_name,reviewsAfter=i[0],reviewsBefore=i[1])
        load_to_sql("REVIEWS_FULL_REFRESH","RCAMP",file_name)

    
    last_review_date='2001-01-01'
    print ("Products " + last_review_date)
    truncate_sql("PRODUCTS","RCAMP")
    file_name = 'C:\\Temp\\Products.csv'
    delete_file(file_name)
    product_extract_api(file_name)
    load_to_sql("PRODUCTS","RCAMP",file_name)

    execute_sql("ReviewCamp_Reviews","RCAMP")
    run_extract()
        
def call_reviews_api(reviewsAfter=''):
    
    if reviewsAfter == '' :
        reviewsAfter=get_last_review_date("REVIEWS","RCAMP")

    print ("Reviews " , reviewsAfter)
    
    start = datetime.datetime.strptime(reviewsAfter, "%Y-%m-%d")

    print ("Process Start Date ", start)
    create_temp_directory()

    file_name = 'C:\Temp\Reviews.csv'
    delete_file(file_name)

    #reviewsAfter=datetime.datetime(dates, 2,1)
    review_extract_api(file_name,start)
    load_to_sql("REVIEWS","RCAMP",file_name)

def call_reviews_api_both_dates(reviewsAfter='',reviewsBefore=''):
    
    if reviewsAfter == '' :
        reviewsAfter=get_last_review_date("REVIEWS","RCAMP")

    print ("Reviews " , reviewsAfter , "  ", reviewsBefore)
    
    start = datetime.datetime.strptime(reviewsAfter, "%Y-%m-%d")
    end = datetime.datetime.strptime(reviewsBefore, "%Y-%m-%d")

    print ("Process Start Date ", start)
    create_temp_directory()

    file_name = 'C:\Temp\Reviews.csv'
    delete_file(file_name)

    #reviewsAfter=datetime.datetime(dates, 2,1)
    review_extract_api(file_name,start,end)
    load_to_sql("REVIEWS","RCAMP",file_name)
    
def read_final_table():
    sql = "SELECT * FROM RCAMP.ALL_REVIEWS"
    cxn = sql_connect_py('HB-SQL-DEV01','HBBCPOPR')
    dfg=trimAllColumns(pd.read_sql_query(sql,cxn))
    cxn.close()
    dfg['Date'] = pd.to_datetime(dfg['Date'], format = '%Y-%m-%d')
    dfg['Prod_Rel_date'] = pd.to_datetime(dfg['Prod_Rel_date'], format = '%Y-%m-%d')
    
    return dfg

def write_extract():
    """
     Item Master Data
    """
    # Get any reviews on the last review date - just to be sure we capture all the records 
    truncate_sql("RCAMP_IIM","REF")
    file_name = 'C:\Temp\RCAMP_IIM_LOAD.csv'
    delete_file(file_name)
    rcamp_iim_load(file_name)
    load_to_sql("RCAMP_IIM","REF",file_name)
    print("Load to ITEM MASTER COMPLETE")
    execute_sql("ReviewCamp_Reviews","RCAMP")
    print("EXEC SP COMPLETE")
    # Refresh extract

# def write_extract_2():
    # tab_file = "R:\\_Department\\Analytics\\reviewCamp\\WorkInProgress\\ReviewCamp.hyper"
    # os.remove(tab_file)
    # pnd.pandleau(read_final_table()).to_tableau(tab_file, add_index=False)

def email_success():
    recipients = [
    #'trevor.downes@hamiltonbeach.com',
    #'riley.irving@hamiltonbeach.com',
     'thomas.loeber@hamiltonbeach.com', 
     'mallik.pullela@hamiltonbeach.com',
    ]
    subject = 'The Review Camp Dashboard has been refreshed.'
    text = '''Please find it at the following location: R:\_Department\Analytics\reviewCamp\WorkInProgress'''
    
    send_email(
        send_to=recipients,
        subject=subject,
        text=text,
    )
    
# def email_failure():
    # recipients = [
    # 'thomas.loeber@hamiltonbeach.com', 
    # # 'mallik.pullela@hamiltonbeach.com',
    # ]
    # subject = 'Error running Review Camp'
    # text = f'The following error was raised: {sys.exc_info()[:2]}'
    
    # send_email(
        # send_to=recipients,
        # subject=subject,
        # text=text,
    # )    
    
def main():
    
    """
     Review  Data  - for daily runs 
    """
    # Get the Last Review load date
    last_review_date=get_last_review_date("REVIEWS","RCAMP")
    print(last_review_date)
    # Get any reviews on the last review date - just to be sure we capture all the records 
    del_sql("REVIEWS","RCAMP",last_review_date)
    call_reviews_api(last_review_date)

    """
     Product Data 
     Do not change the Hard Coded Last Review Date 
    """
    last_review_date='2001-01-01'
    print ("Products " + last_review_date)
    truncate_sql("PRODUCTS","RCAMP")
    file_name = 'C:\\Temp\\Products.csv'
    delete_file(file_name)
    product_extract_api(file_name)
    load_to_sql("PRODUCTS","RCAMP",file_name)

    execute_sql("ReviewCamp_Reviews","RCAMP")
    
    tab_file = "R:\\_Department\\Analytics\\reviewCamp\\WorkInProgress\\ReviewCamp.hyper"
    #pandleau(read_final_table()).to_tableau(tab_file, add_index=False)
    
    tableau_utils.write_extract(
        read_final_table(),
        tab_file,
        #"R:\\_Department\\Analytics\\reviewCamp\\WorkInProgress\\ReviewCamp.hyper",
        False
    )
    
    # Notify customers of update
    email_success()    
    
    print ('Now 4= ', datetime.datetime.now(), 'Complete')

def run_extract():
    tab_file = "R:\\_Department\\Analytics\\reviewCamp\\WorkInProgress\\ReviewCamp.hyper"
    #pandleau(read_final_table()).to_tableau(tab_file, add_index=False)
    
    tableau_utils.write_extract(
        read_final_table(),
        tab_file,
        #"R:\\_Department\\Analytics\\reviewCamp\\WorkInProgress\\ReviewCamp.hyper",
        False
    )
    
    # Notify customers of update
    email_success()        

if __name__ == '__main__':
    print ('Now 1 = ', datetime.datetime.now())
    
    main()  # this for a daily run
    
    # To run a full refresh - comment the "above line and  run the below line 
    # reviewsbefore - Add this if you want to limit the extract 
    #complete_refresh(reviewsafter='2001-01-01',reviewsbefore='')  # this for a daily run
    
    
    print ('Now 4= ', datetime.datetime.now(), 'Complete')
