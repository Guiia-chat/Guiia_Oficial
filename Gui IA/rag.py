# app.py

# --- Imports Básicos ---
import os
import re 
from pathlib import Path 
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import time 

# --- Imports do LangChain ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Certifique-se que pypdf está instalado: pip install pypdf
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict

# --- 1. CARREGAMENTO DE VARIÁVEIS E MODELO ---
print("Carregando variáveis de ambiente...")
load_dotenv()
GOOGLE_API_KEY = os.getenv('API_KEY')

print("Inicializando o modelo de IA (LLM)...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)
print("✅ Modelo de IA (LLM) carregado com sucesso!")

print("Inicializando o modelo de Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings( 
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
print("✅ Modelo de Embeddings carregado com sucesso!")


# --- 2. FUNÇÕES AUXILIARES ---
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

def format_docs(docs: List) -> str:
    """Converte a lista de documentos em uma única string de texto."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 3. SETUP DO PIPELINE RAG (INTELIGENTE) ---
print("Iniciando o pipeline de RAG...")

retriever = None
rag_chain = None
FAISS_INDEX_PATH = "faiss_index_store"  # Nome da pasta para salvar o banco

def setup_rag_pipeline():
    global retriever, rag_chain 

    # --- VERIFICAÇÃO DE CACHE (SALVA TEMPO E COTA) ---
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"💾 Banco de dados encontrado em '{FAISS_INDEX_PATH}'! Carregando do disco...")
        try:
            # Carrega o banco salvo sem gastar API
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ Banco carregado do disco com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar do disco: {e}. Vamos recriar.")
            # Se der erro ao carregar, apaga a variável para recriar abaixo
            vectorstore = None
    else:
        vectorstore = None

    # --- SE NÃO CARREGOU DO DISCO, CRIA DO ZERO ---
    if vectorstore is None:
        print("📂 Criando banco de dados do zero (lendo PDFs)...")
        pdf_directory = Path(r"C:/Users/mathe/Downloads/Gui-IA-Oficial-main/Gui IA/documents")

        
        docs = []
        print(f"Procurando PDFs em: {pdf_directory}")
        for n in pdf_directory.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(n)) 
                docs.extend(loader.load())
                print(f"Carregado: {n.name}")
            except Exception as e:
                print(f"Erro ao carregar {n.name}: {e}")
        
        if not docs:
            print("ALERTA: Nenhum PDF encontrado.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)
        print(f"Total de {len(chunks)} chunks para processar.")

        print("Gerando Embeddings em lotes (para evitar erro de Cota 429)...")
        try:
            # 1. Cria o primeiro lote
            vectorstore = FAISS.from_documents([chunks[0]], embeddings)
            
            # 2. Adiciona o resto com pausas
            batch_size = 5
            for i in range(1, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                print(f"Processando {i}/{len(chunks)}...")
                vectorstore.add_documents(batch)
                time.sleep(5) # Pausa de 5s é suficiente geralmente

            # 3. SALVA NO DISCO PARA A PRÓXIMA VEZ
            print("💾 Salvando banco no disco...")
            vectorstore.save_local(FAISS_INDEX_PATH)
            print("✅ Banco salvo com sucesso!")

        except Exception as e:
            print(f"Erro crítico na criação do banco: {e}")
            return

    # --- CRIAÇÃO DA CHAIN (IGUAL ANTES) ---
    if vectorstore:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 4}
        )

        prompt_rag = ChatPromptTemplate.from_messages([
            ("system",
             "Você é um Assistente de uma universidade e ajuda alunos com as materias. "
             "Tente identificar capitulos, topicos, sub-topicos e onde se localizam. "
             "Responda SOMENTE com base no contexto fornecido. Seja claro. "
             "Se for solicitado explique com detalhes. "
             "Caso pergunte algo fora do contexto mas sobre tecnologia, explique basicamente e avise que não está na apostila. "
             "Se a resposta não estiver no contexto, responda apenas 'Desculpe, não encontrei essa informação nos meus documentos.'"),
            ("human", "Pergunta: {question}\n\nContexto:\n{context}")
        ])
        
        rag_chain = (
            RunnableMap({
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": RunnablePassthrough.assign(question=lambda x: x["question"])
            })
            | prompt_rag
            | llm
            | StrOutputParser()
        )
        
        print("✅ Chain de RAG (rag_chain) pronta!")

# Executa o setup ao iniciar
setup_rag_pipeline()

def responder_pergunta(question: str):
    if rag_chain is None:
        return "Erro interno: RAG não inicializado."

    try:
        resposta = rag_chain.invoke({"question": question})
        return resposta
    except Exception as e:
        print("Erro ao gerar resposta:", e)
        return "Erro ao processar sua pergunta."
    
