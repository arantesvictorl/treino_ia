from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

PROMPT_TEMPLATE = """
Idade: {Idade}
Altura: {Altura}
Peso: {Peso}
Frequência: {Frequência}
Sexo: {Sexo}
Grau: {Grau}
Objetivo: {Objetivo}
Tem Restrição: {Tem_Restrição}
Com base nesse template, forneça um plano de treino de musculação personalizado com as fichas (A,B,C ou mais dependendo do objetivo e grau) para o usuário, sendo cada ficha para um dia diferente. Considere as informações específicas inseridas nos campos do template para ajustar o treino de acordo com as necessidades e objetivos do usuário, garantindo um plano bem alinhado às suas características e metas de saúde e fitness. Lembre que a saída deve ser apenas as fichas de treino e nada mais. Nas fichas faça a divisão entre exercicios de Costas, Peito, Quadriceps, Posterior, Panturrilha, Triceps, Biceps, Abdomen... tente equilibrar da melhor maneira possível esses grupamentos musculares nas N fichas que você decidir. Não se esqueça de incluir o número de séries e repetições para cada exercício. NÃO COLOQUE EXERCÍCIOS DE SUPERIORES COM INFERIORES NO MESMO DIA.
"""
TEMPERATURE = 0.2
MAX_TOKENS = 480
def main():


    st.title('Formulário de Anamnese')

    idade = st.number_input("Idade", min_value=0, max_value=120, step=1)
    altura = st.number_input("Altura (em cm)", min_value=50, max_value=300, step=1)
    peso = st.number_input("Peso (em kg)", min_value=1, max_value=300, step=1)
    frequencia = st.selectbox("Frequência de atividades na semana", ['1X', '2X', '3X', '4X', '5X', '6X'])
    sexo = st.radio("Sexo", ['Feminino', 'Masculino', 'Outro'])
    grau = st.selectbox("Grau de treino", ['Iniciante', 'Intermediário', 'Avançado'])
    objetivo = st.selectbox("Objetivo", ['Emagrecimento', 'Ganho de Massa', 'Qualidade de Vida', 'Controlar Doença'])
    tem_restricao = st.text_area("Tem restrição médica? Se sim, quais?")

    if st.button('Predizer Modelo'):
        anamnese = {
            "Idade": idade,
            "Altura": altura,
            "Peso": peso,
            "Frequência": frequencia,
            "Sexo": sexo,
            "Grau": grau,
            "Objetivo": objetivo,
            "Tem Restrição": tem_restricao
        }

        anamnese_str = PROMPT_TEMPLATE.format(Idade=anamnese["Idade"], Altura=anamnese["Altura"], Peso=anamnese["Peso"], Frequência=anamnese["Frequência"], Sexo=anamnese["Sexo"], Grau=anamnese["Grau"], Objetivo=anamnese["Objetivo"], Tem_Restrição=anamnese["Tem Restrição"])
        llm = ChatGoogleGenerativeAI(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, model="gemini-pro")
        prompt = PromptTemplate(input_variables=list(anamnese.keys()), template=anamnese_str)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.invoke(anamnese)
        st.write(response["text"])

if __name__ == "__main__":
    main()
