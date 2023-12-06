import re
import sys

def extract_info(text):
    
    user_name    = None
    user_email   = None
    user_cpf     = None
    user_cnumber = None
    
    name_match = re.search(r"\b(?:[A-Z][a-zA-ZÀ-ÿ'-]*(?:\s(?:d[aeio]s?\s)?[A-Z][a-zA-ZÀ-ÿ'-]*)+)\b", text)
    if name_match:
        user_name = name_match.group()
    else:
        user_name = "Não reconhecido"

    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    if email_match:
        user_email = email_match.group()
    else:
        user_email = "Não reconhecido"

    cpf_match = re.search(r"\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11}|\d{3}\ \d{3}\ \d{3} \d{2})\b", text)
    if cpf_match:
        user_cpf = cpf_match.group()
    else:
        user_cpf= "Não reconhecido"
    
    cnumber_match = re.search(r"\b(?:\(\d{2}\)\s\d{5}-\d{4}|\d{2}\s\d{5}\s\d{4}|\d{2}\s\d{9})\b|\d{2}\s\d{5}-\d{4}", text)
    if cnumber_match:
        user_cnumber = cnumber_match.group()
    else:
        user_cnumber = "Não reconhecido"

    return user_name, user_email, user_cpf, user_cnumber


def finalizar_atendimento():
    print("Deseja continuar com o atendimento?\n1 - Sim\n2 - Não")
    opcao = input("\n> ").lower()

    padrao_nao = r'n(ao|ã[o])?|n'
    padrao_sim = r'sim|s' 

    nao = re.search(padrao_nao, opcao)
    sim = re.search(padrao_sim, opcao)

    if nao:
        print("Entendo. Se precisar de assistência no futuro, não hesite em entrar em contato novamente. Tenha um bom dia!")
        sys.exit()
    if sim:
        print("\nEm que mais posso ajudar?")

    
    

def protocolo_cadastro():
    print("Será necessário preencher formulário físico com os dados do usuário, assinado pelo secretário e encaminhar para o setor de Tecnologia de Informação.\nSegue o formulário para preenchimento.")
    print("[FORMULÁRIO ENVIADO]") # Função para envio de documentos
    finalizar_atendimento()


def protocolo_login():
    print("Entendido, para ser aberto o chamado será necessário das seguintes informações:")
    print("Nome;\nCPF;\nEmail Pessoal (ou Institucional);\nTelefone.")
    name, email, cpf, number = extract_info(input("> "))
    print(f"[Dados reconhecidos:]\n[Nome: {name}],\n[CPF: {cpf}],\n[Email: {email}],\n[Número: {number}],\n\nQual seria o problema em questão?")
    problem = input("> ")
    print("[",problem,"]")
     # abrir_chamado(name,email,cpf,number,problem)
    print("Sua solicitação foi registrada e encaminhada a um técnico responsável para que seja devidamente solucionada.")
    finalizar_atendimento()