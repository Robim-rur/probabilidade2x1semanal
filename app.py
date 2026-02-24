import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Prob2x1 – Scanner Estatístico Semanal (+2% / -1%)")

st.markdown("""
Este aplicativo executa um estudo estatístico de curto prazo,
avaliando a probabilidade histórica de um ativo atingir +2% de valorização
antes de sofrer uma queda de −1%, dentro de uma janela de até 5 pregões.

O estudo é puramente quantitativo e não utiliza setups gráficos,
indicadores técnicos ou regras operacionais.
""")

# =========================================================
# PARÂMETROS FIXOS
# =========================================================

TARGET = 0.02
STOP = -0.01
HORIZON = 5

# =========================================================
# LISTA FIXA DE ATIVOS (178)
# =========================================================

ativos_scan = sorted(set([
    "RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
    "BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
    "COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
    "ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
    "HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
    "MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
    "RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
    "UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA","AURE3.SA","BHIA3.SA","CASH3.SA",
    "CVCB3.SA","DIRR3.SA","ENAT3.SA","GMAT3.SA","IFCM3.SA","INTB3.SA","JHSF3.SA","KEPL3.SA","MOVI3.SA","ORVR3.SA",
    "PETZ3.SA","PLAS3.SA","POMO4.SA","POSI3.SA","RANI3.SA","RAPT4.SA","STBP3.SA","TEND3.SA","TUPY3.SA",
    "BRSR6.SA","CXSE3.SA","AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA",
    "NVDC34.SA","MELI34.SA","BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA",
    "WMTB34.SA","NIKE34.SA","ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA",
    "HDCO34.SA","INTC34.SA","JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA",
    "PFIZ34.SA","PMIC34.SA","QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA",
    "VZUA34.SA","ABTT34.SA","AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA","BOVA11.SA","IVVB11.SA",
    "SMAL11.SA","HASH11.SA","GOLD11.SA","GARE11.SA","HGLG11.SA","XPLG11.SA","VILG11.SA","BRCO11.SA","BTLG11.SA",
    "XPML11.SA","VISC11.SA","HSML11.SA","MALL11.SA","KNRI11.SA","JSRE11.SA","PVBI11.SA","HGRE11.SA","MXRF11.SA",
    "KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"
]))

st.write(f"Quantidade de ativos na base: {len(ativos_scan)}")

anos = st.slider("Anos de histórico para o estudo", 5, 12, 10)

# =========================================================
# FUNÇÃO DE ESTUDO
# =========================================================

def estudar_ativo(ticker, anos):

    fim = datetime.today()
    inicio = fim - timedelta(days=365 * anos)

    df = yf.download(
        ticker,
        start=inicio.strftime("%Y-%m-%d"),
        end=fim.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )

    if df.empty or len(df) < 200:
        return None

    df = df.dropna()

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    total_janelas = 0
    wins = 0
    losses = 0

    max_retorno_lista = []
    max_drawdown_lista = []

    for i in range(len(df) - HORIZON - 1):

        entrada = closes[i]
        alvo = entrada * (1 + TARGET)
        stop = entrada * (1 + STOP)

        ordem = None
        max_ret = -999.0
        max_dd = 999.0

        for j in range(1, HORIZON + 1):

            h = highs[i + j]
            l = lows[i + j]

            r_max = (h / entrada) - 1
            r_min = (l / entrada) - 1

            max_ret = max(max_ret, r_max)
            max_dd = min(max_dd, r_min)

            if h >= alvo and l <= stop:
                ordem = "ambos"
                break
            if h >= alvo:
                ordem = "alvo"
                break
            if l <= stop:
                ordem = "stop"
                break

        total_janelas += 1
        max_retorno_lista.append(max_ret)
        max_drawdown_lista.append(max_dd)

        if ordem == "alvo":
            wins += 1
        elif ordem == "stop":
            losses += 1

    if total_janelas == 0:
        return None

    prob_win = wins / total_janelas
    prob_loss = losses / total_janelas

    retorno_medio_max = np.mean(max_retorno_lista)
    drawdown_medio = np.mean(max_drawdown_lista)

    # Expectativa matemática usando exatamente +2% e -1%
    expectativa = prob_win * TARGET + prob_loss * STOP

    # Score de lucratividade ajustado por risco e potencial
    score = expectativa * (retorno_medio_max / TARGET)

    ult_close = closes[-1]

    return {
        "Ativo": ticker.replace(".SA", ""),
        "Amostras": total_janelas,
        "Prob_Gain_2pct": prob_win,
        "Prob_Loss_1pct": prob_loss,
        "Expectativa": expectativa,
        "Retorno_max_médio": retorno_medio_max,
        "Drawdown_médio": drawdown_medio,
        "Score_classificacao": score,
        "Preço_atual": ult_close
    }

# =========================================================
# EXECUÇÃO
# =========================================================

if st.button("Rodar estudo estatístico"):

    resultados_finais = []

    with st.spinner("Processando ativos..."):

        for t in ativos_scan:
            try:
                r = estudar_ativo(t, anos)
                if r is not None:
                    resultados_finais.append(r)
            except Exception:
                pass

    if len(resultados_finais) == 0:
        st.warning("Nenhum ativo retornou dados suficientes.")
    else:

        df = pd.DataFrame(resultados_finais)

        df["Prob_Gain_2pct_%"] = df["Prob_Gain_2pct"] * 100
        df["Prob_Loss_1pct_%"] = df["Prob_Loss_1pct"] * 100
        df["Expectativa_%"] = df["Expectativa"] * 100
        df["Retorno_max_médio_%"] = df["Retorno_max_médio"] * 100
        df["Drawdown_médio_%"] = df["Drawdown_médio"] * 100

        abas = st.tabs(["Tabela completa", "Ranking estatístico (Top 5)"])

        with abas[0]:

            tabela = df[[
                "Ativo","Amostras",
                "Prob_Gain_2pct_%","Prob_Loss_1pct_%",
                "Expectativa_%","Retorno_max_médio_%",
                "Drawdown_médio_%","Score_classificacao","Preço_atual"
            ]].sort_values(by="Score_classificacao", ascending=False)

            tabela = tabela.rename(columns={
                "Prob_Gain_2pct_%": "Prob. bater +2% (%)",
                "Prob_Loss_1pct_%": "Prob. bater -1% (%)",
                "Expectativa_%": "Expectativa matemática (%)",
                "Retorno_max_médio_%": "Máx. retorno médio (%)",
                "Drawdown_médio_%": "Drawdown médio (%)",
                "Score_classificacao": "Score estatístico"
            })

            st.subheader("Resultados completos")
            st.dataframe(tabela, use_container_width=True)

            csv = tabela.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar tabela completa",
                csv,
                "resultado_prob2x1_completo.csv",
                "text/csv"
            )

        with abas[1]:

            ranking = df.sort_values(
                by="Score_classificacao",
                ascending=False
            ).head(5)

            ranking = ranking[[
                "Ativo",
                "Prob_Gain_2pct_%",
                "Prob_Loss_1pct_%",
                "Expectativa_%",
                "Retorno_max_médio_%",
                "Drawdown_médio_%",
                "Score_classificacao"
            ]]

            ranking = ranking.rename(columns={
                "Prob_Gain_2pct_%": "Prob. +2% (%)",
                "Prob_Loss_1pct_%": "Prob. -1% (%)",
                "Expectativa_%": "Expectativa (%)",
                "Retorno_max_médio_%": "Máx. retorno médio (%)",
                "Drawdown_médio_%": "Drawdown médio (%)",
                "Score_classificacao": "Score estatístico"
            })

            st.subheader("Top 5 – melhores ativos pelo critério estatístico")

            st.dataframe(ranking, use_container_width=True)
