
from testefask import app
from flask import render_template, request, jsonify
from fine_tuning import gerar_resposta
from rag import responder_pergunta  
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/Login')
def Login():
    return render_template('Login.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
    
@app.route('/sobre_gui')
def sobre_gui():
    return render_template('sobre_gui.html')

@app.route('/sobre_nos')
def sobre_nos():
    return render_template('sobre_nos.html')    

@app.route('/ajuda_guiia')
def ajuda_guiia():
    return render_template('ajuda_guiia.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    data = request.get_json()
    pergunta = data.get('pergunta')

    if not pergunta:
        return jsonify({'resposta': 'Pergunta vazia'})

    resposta = gerar_resposta(pergunta)
    return jsonify({'resposta': resposta})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error' : 'Pergunta vazia'}), 400

    resposta = responder_pergunta(question)
    return jsonify({'answer': resposta})
