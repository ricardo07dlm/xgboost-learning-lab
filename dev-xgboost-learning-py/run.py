import uvicorn

# USE ONLY EN LOCAL


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run("src.api.router:app", host="0.0.0.0", port=8001, reload=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
