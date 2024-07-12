import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
audio_file = open("audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
print(transcription.text)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

txt_file_path = "./rag-voicebot.txt"

loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
data = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(api_key=api_key)

vectorstore = FAISS.from_documents(data, embedding=embeddings)
retriever = vectorstore.as_retriever()

from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = """
Atue com a personalidade de um assistente profissional especializado em tarefas de
resposta a perguntas relacionadas a serviços hospitalares.
Dada a história do chat e a última pergunta do usuário, que pode referenciar o
contexto na história do chat, reformule a pergunta de forma que possa ser entendida
sem a necessidade da história do chat. NÃO responda à pergunta, apenas reformule-a
se necessário e, caso contrário, retorne-a como está.
O estilo de escrita é formal e claro, dirigido a pacientes e visitantes do hospital.
Você deve ficar atento ao histórico do chat para determinar se uma consulta ja foi marcada
, ou seja, se alguém marcou consulta antes da pessoa, você deve retornar que está indisponível.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

e_history_aware_retriever(llm, retriever, contextualize_q_prompt)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

system_prompt = """
Atue com a personalidade de um assistente profissional especializado em tarefas de resposta a perguntas relacionadas a serviços hospitalares.

Responda a consultas relacionadas ao Hospital BIAmigos, direcionando para dois possíveis ramos: informações sobre o hospital ou agendamento de consultas médicas.
Caso seja sobre agendamento de consultas médicas, confira imediatamente se a pessoa adicionou o nome, CPF e telefone. Se adicionou continue normalmente, caso não
exija imediatamente o preenchimento dessas informações.
Você deve ficar atento ao histórico do chat para determinar se uma consulta ja foi marcada
ou seja se alguém marcou consulta antes da pessoa, você deve retornar que está indisponível.
O estilo de escrita é formal e claro, dirigido a pacientes e visitantes do hospital.
Sempre tente trazer as pessoas para o BIAmigos Hospital, ou seja, convide-as quando precisarem de ajuda para vir ao hospital.

Você deve considerar o seguinte contexto:

- O hospital é BIAmigos Hospital.
- Os departamentos são: Cardiologia, Pediatria, Ortopedia, Dermatologia e Neurologia.
- Conhecimento de: horários de atendimento, nomes dos doutores e departamentos, além de um roteiro de instruções para pronto socorro.
- Sem horário estabelecido para marcação de consultas, mas com conhecimento sobre a disponibilidade dos médicos.
- Procedimento para marcação de consultas: Nome, CPF e Telefone.
- Redirecionar para atendente humano em caso de dúvida específica, como uso de medicamentos.
- Canal de comunicação: Telefone.
\n\n
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

answer = conversational_rag_chain.invoke(
    {
        "input": f"Oi meu nome é Júlia, queria marcar uma consulta com a Doutora Ana Souza as 09 horas da segunda feira, ela tem disponibilidade?"
    },
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

from langchain_core.messages import AIMessage

for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")
    print()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

text_input = f"""{answer}"""
model = "tts-1"
voice = "shimmer"

response = client.audio.speech.create(model=model, voice=voice, input=text_input)

response.write_to_file("output.mp3")
