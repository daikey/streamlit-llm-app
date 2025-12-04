import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def call_llm(user_input: str, expert_type: str) -> str:
    # OpenAI の LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",   
        temperature=0.7
    )

    # プロンプトテンプレート
    template = """
あなたは{expert_type}です。
ユーザーからの相談に対して、専門家としてわかりやすく、具体的に助言してください。

# 入力
{user_input}
"""

    prompt = PromptTemplate(
        input_variables=["expert_type", "user_input"],
        template=template
    )

    # LangChain Expression Language（パイプ構文）
    chain = prompt | llm | StrOutputParser()

    # パラメータを渡して実行
    response = chain.invoke({
        "expert_type": expert_type,
        "user_input": user_input
    })

    return response


def main():
    st.title("専門家アドバイス生成アプリ")

    st.write(
        """
このアプリでは、入力したテキストを **LangChain** 経由で LLM に送り、  
選択した専門家の視点で回答を生成します。
        """
    )

    st.divider()

    # ラジオボタン：専門家選択
    expert_choice = st.radio(
        "相談したい専門家を選んでください：",
        ["マーケティング専門家", "Webアプリのエンジニア", "UXデザイナー"]
    )

    # テキスト入力
    user_input = st.text_area(
        "相談内容を入力してください：",
        height=150,
        placeholder="例：新サービスのリテンションを高めたい。改善点は？"
    )

    if st.button("送信"):
        if not user_input.strip():
            st.warning("相談内容を入力してください。")
        else:
            with st.spinner("回答を生成中..."):
                answer = call_llm(user_input, expert_choice)
                st.subheader("専門家からの回答")
                st.write(answer)


if __name__ == "__main__":
    main()
