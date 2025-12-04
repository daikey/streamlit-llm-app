import os
from dotenv import load_dotenv

import streamlit as st
from langchain.openai import chatopenai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

def call_llm(user_input: str, expert_type: str) -> str:
    llm = chatopenai.ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    prompt_template = """
    あなたは{expert_type}です。
    ユーザーからの相談に対して、専門家としてわかりやすく、具体的なアドバイスを日本語で答えてください。
    # ユーザーからの入力
    {user_input}
    """

    prompt = PromptTemplate(
        input_variables=["expert_type", "user_input"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(user_input)

    return response


def main():
    st.title("専門家アドバイス生成アプリ")

    st.write(
        """
        このアプリでは、入力したテキストを **LangChain** 経由で LLM に送り、選択した「専門家」の視点で回答を表示します。
        
        1. 下のラジオボタンから、どんな専門家に相談したいかを選びます  
        2. テキスト入力欄に質問や相談内容を入力します  
        3. 「送信」ボタンを押すと、LLMからの回答が表示されます
        """
    )
    st.divider()

    # 専門家タイプをラジオボタンで選択
    expert_choice = st.radio(
        "相談したい専門家のタイプを選んでください。",
        ["マーケティング・CRMの専門家", "webアプリの専門家", "UXエンジニア"]
    )

    expert_role = expert_choice
    user_input = st.text_area(label="相談内容を入力してください。",height=150,placeholder="例: 新しい製品のマーケティング戦略についてアドバイスが欲しいです。")

    if st.button("送信"):
        if not user_input.strip():
            st.warning("相談内容を入力してください。")
        else:
            with st.spinner("アドバイスを生成中..."):
                try:
                    answer = call_llm(user_input, expert_type=expert_role)
                    st.divider()
                    st.subheader("専門家からのアドバイス")
                    st.write(answer)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")  

if __name__ == "__main__":
    main()