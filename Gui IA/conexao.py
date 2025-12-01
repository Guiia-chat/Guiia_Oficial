##usei a biblioteca mysql-connector-python através do comando pip install mysql-connector-python(slc nunca tinha mexido nisso antes e aprendi sem curso nem nada kkkkkkk)
import pymysql


conexao = pymysql.connect(
    host='localhost',
    user='root',
    password='1507',
    database=' Chat_bot_Guiia'
)
cursor = conexao.cursor()
cursor.execute("SELECT * FROM Chat_bot")
for linha in cursor.fetchall():
    print(linha)##fetchone() pega só a primeira linha, fetchall() pega todas as linhas, o fecthall() retorna uma lista com todas as linhas, então é necessário fazer um for para printar cada linha inclusive o facthall passa no formato que é mostrado no banco de dados, ou seja, uma tupla

