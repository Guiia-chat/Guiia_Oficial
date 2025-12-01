from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymysql

# Variáveis globais para armazenar o modelo e a base de dados
# O modelo é carregado APENAS UMA VEZ quando este arquivo é importado
MODELO = SentenceTransformer('all-MiniLM-L6-v2') 
EMBEDDINGS_BD = None
RESPOSTAS_BD = None

# THRESHOLD: Valor mínimo de similaridade para considerar a resposta como "válida"
THRESHOLD = 0.45 


def inicializar_banco_de_dados(host, user, password, database):
    """Carrega dados do BD, gera embeddings e os armazena globalmente."""
    global EMBEDDINGS_BD, RESPOSTAS_BD
    
   
    conexao = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conexao.cursor()
    cursor.execute("SELECT pergunta, resposta FROM Chat_bot")
    dados = cursor.fetchall()
    cursor.close()
    conexao.close()
    
    # 2. Processamento dos Dados
    perguntas = [item[0] for item in dados]
    RESPOSTAS_BD = [item[1] for item in dados]
    
    # 3. Geração dos Embeddings (Vetores)
    EMBEDDINGS_BD = MODELO.encode(perguntas, convert_to_tensor=False)
    print(" Base de dados e embeddings carregados com sucesso!")


def buscar_resposta_semelhante(pergunta_usuario):
    """Calcula a similaridade e retorna a melhor resposta ou uma mensagem padrão."""
    global THRESHOLD
    
    if EMBEDDINGS_BD is None:
        return "Erro: O sistema de busca não foi inicializado."
    
    
    embedding_pergunta_usuario = MODELO.encode([pergunta_usuario], convert_to_tensor=False)

    # 2. Calcula a similaridade (Sklearn)
    similaridades = cosine_similarity(embedding_pergunta_usuario, EMBEDDINGS_BD)
    
    # 3. Encontra o índice e o valor da maior similaridade
    indice_resposta_mais_semelhante = np.argmax(similaridades)
    resposta_mais_semelhante = similaridades[0][indice_resposta_mais_semelhante] 
    
    # 4. Lógica de Decisão (usando o THRESHOLD)
    print(f"Similaridade calculada: {resposta_mais_semelhante:.4f}")

    if resposta_mais_semelhante < THRESHOLD:
        return "Desculpe, não entendi sua pergunta. Tente reformular."
    else:
        return RESPOSTAS_BD[indice_resposta_mais_semelhante]
