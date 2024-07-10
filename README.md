# VoiceBOT
por David O'Neil, Carlos Eduardo Rocha, Gabriel Orlow, José Ricardo Fleury e Lucas Wanderley

## Introdução
Chatbot desenvolvido para o Projeto FMF, ministrado pelo Professor Federson da UFG. O foco é apresentar uma ideia original à aquela apresentada pelo professor em sala. a inspiração para o projeto veio do modelo gerado em https://github.com/jerrytigerxu/Simple-Python-Chatbot.

## AGEMC do Projeto FMF
Depois de escolhido qual o projeto que seria o molde da aplicação, foi pensado em como pode-se categorizar esse modelo nas diferentes letras do AGEMC, determinado pelo grupo seguindo as respostas abaixo:

### A ou PERGUNTA
Dada a base de treinamento do modelo, a query do usuário e as etapas de pós-processamento do contexto, qual seriam as possibilidades de resposta do chatbot?

### GE ou EXPLORAÇÃO
As técnicas utilizadas foram lemmatizer, tokenize, enumerate, sequential.

### M ou MODELAGEM DAS TÉCNICAS
São usadas algumas funções após o uso do modelo Sequential do Keras, que consiste na rede neural multi-layer perceptron. Essas funções tem como objetivo modelar o output do modelo para a visualização na interface de front criada. Elas são: 
 - `clean_up_sentence()` : limpa os inputs
 - `bow()` :  pega os inputs limpos e os seleciona em um conjunto de palavras que será usado para a predição das classes treinadas
 - `predict_class()` : através de um threshold de 0,25 (para evitar overfitting), cria um output através da probabilidade de intents se relacionarem ao treinamento
 - `chatbot_response()` : recebe a mensagem do usuário e o classifica através do `predict_class` e produz um output relacionado ao treinamento através do `getResponse()`

### C ou VISUALIZAÇÃO
Para visualizar a resposta dos outputs gerados nas funções anteriores, a biblioteca tkinker é usada para gerar uma GUI em Python. Assim uma interface para a comunicação Q/A entre o usuário e o modelo é criado, gerando um streaming de mensagens de output
