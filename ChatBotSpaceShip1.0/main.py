import openai

# Defina a sua chave de API do OpenAI
openai.api_key = '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'

# Função para enviar uma mensagem para o chatbot
def enviar_mensagem(mensagem, chat_history=[]):
    # Define o modelo que você deseja usar
    modelo = 'gpt-3.5-turbo'

    # Cria a estrutura do chat
    chat = [
        {'role': 'system', 'content': 'Você é um chatbot feito para o setor de HelpDesk da prefeitura de Macaé, desenvolvido pelo Lorde Supremo Eduardo'},
        {'role': 'user', 'content': mensagem}
    ]
    chat.extend(chat_history)

    # Chama a API do OpenAI para gerar uma resposta de chat
    resposta = openai.ChatCompletion.create(
        model=modelo,
        messages=chat,
        max_tokens=1000,
        temperature=0.7
    )

    # Retorna a resposta gerada pela API
    return resposta.choices[0].message.content.strip()

# Loop principal do chatbot
print('Bem-vindo ao Chatbot! Digite "sair" para encerrar o chat.')
chat_history = []
while True:
    # Lê a entrada do usuário
    entrada = input('Usuário: ')

    # Verifica se o usuário quer sair
    if entrada.lower() == 'sair':
        break

    # Envia a mensagem para o chatbot e imprime a resposta
    resposta = enviar_mensagem(entrada, chat_history)
    chat_history.append({'role': 'user', 'content': entrada})
    chat_history.append({'role': 'system', 'content': resposta})
    print('Chatbot:', resposta)
