import pymysql
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("API_KEY")

def get_database_connection():   
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="1507",
        database="Chat_bot_Guiia"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT pergunta,resposta FROM Chat_bot")
    dados = cursor.fetchall()
    cursor.close()
    connection.close()    
    
    contexto = "\n".join([
        f"Pergunta: {pergunta}\nResposta: {resposta}"
        for pergunta, resposta in dados
    ])
    return contexto


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.0
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """
Você é um assistente útil. Use o contexto para responder detalhadamente:

Instruções:
1. Use o contexto abaixo para formular respostas explicativas.
2. Conecte informações relacionadas sempre que possível.
3. Evite respostas curtas; seja completo e claro.
4. Tome um tom natural e informativo.

Contexto:
{contexto}
     """
    ),
    ("human", "{pergunta_usuario}")
])
def gerar_resposta(pergunta_usuario):
    contexto = get_database_connection()
    
    # 1. Cria a Chain combinando Prompt e LLM
    chain = prompt_template | llm | StrOutputParser() # Adicione o parser para garantir o texto limpo
    
    # 2. Invoca a Chain passando o dicionário de inputs
    response = chain.invoke({
        "contexto": contexto,
        "pergunta_usuario": pergunta_usuario
    })
    # O StrOutputParser() garante que a saída já é uma string
    return response
