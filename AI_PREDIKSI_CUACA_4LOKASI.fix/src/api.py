import requests


API_URL = "https://garamina.com/fintech2/integrasi/dashboard_dir/prodBB_dir/get_data_phpl_cuaca"


for method in ["get", "post"]:

    try:

        if method == "get":
            r = requests.get(API_URL)

        else:
            r = requests.post(API_URL)


        print("================")
        print("METHOD :", method)
        print("STATUS :", r.status_code)
        print(r.text[:500])


    except Exception as e:
        print(e)