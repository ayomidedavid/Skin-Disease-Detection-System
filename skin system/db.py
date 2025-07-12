import pymysql

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        db='skin_disease_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
