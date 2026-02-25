import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Prob2x1 – Scanner Estatístico Semanal (+2% / -1%)")

st.markdown("""
Este aplicativo realiza um estudo estatístico sobre ativos da B3 e BDRs,
avaliando a probabilidade histórica de um ativo atingir +2% de valorização
antes de sofrer uma queda de −1%, dentro de uma janela máxima de 5 pregões.

O modelo é puramente estatístico e não utiliza setups gráficos.
""")

# ============================================================
# LISTA FIXA DOS ATIVOS (178)
# ============================================================

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

# ============================================================
# PARÂMETROS FIXOS DO ESTUDO
# ============================================================

TARGET = 0.02
STOP = 0.01
MAX_DAYS = 5
YEARS_BACK = 10

# ============================================================
# FUNÇÃO DE SIMULAÇÃO
# ============================================================

def simular_ativo(ticker):

    end = datetime.today()
    start = end - timedelta(days=YEARS_BACK * 365)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )
    except:
        return None

    if df is None or len(df) < 50:
        return None

    df = df.dropna()

    wins = 0
    losses = 0
    trades = 0

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    for i in range(len(df) - MAX_DAYS - 1):

        entry = closes[i]

        alvo = entry * (1 + TARGET)
        stop = entry * (1 - STOP)

        resultado = None

        for j in range(1, MAX_DAYS + 1):

            idx = i + j

            if highs[idx] >= alvo and lows[idx] <= stop:
                # empate intrabar -> considera stop primeiro (pior cenário)
                resultado = "loss"
                break

            if highs[idx] >= alvo:
                resultado = "win"
                break

            if lows[idx] <= stop:
                resultado = "loss"
                break

        if resultado == "win":
            wins += 1
            trades += 1
        elif resultado == "loss":
            losses += 1
            trades += 1

    if trades == 0:
        return None

    p_win = wins / trades
    p_loss = losses / trades

    expectancy = (p_win * TARGET) - (p_loss * STOP)

    if p_loss > 0:
        profit_factor = (p_win * TARGET) / (p_loss * STOP)
    else:
        profit_factor = np.inf

    return {
        "Ativo": ticker.replace(".SA", ""),
        "Trades": trades,
        "p_win": p_win,
        "p_loss": p_loss,
        "Expectancy": expectancy,
        "Profit_Factor": profit_factor
    }


# ============================================================
# EXECUÇÃO
# ============================================================

if st.button("Rodar estudo estatístico"):

    resultados = []

    barra = st.progress(0)

    for i, ticker in enumerate(ativos_scan):
        r = simular_ativo(ticker)
        if r:
            resultados.append(r)
        barra.progress((i + 1) / len(ativos_scan))

    if len(resultados) == 0:
        st.error("Nenhum ativo retornou dados válidos.")
        st.stop()

    df = pd.DataFrame(resultados)

    # ============================================================
    # FILTRO OBRIGATÓRIO
    # ============================================================

    df_filtrado = df[df["p_win"] > df["p_loss"]].copy()

    # ============================================================
    # RANKING FINAL
    # ============================================================

    df_ranking = df_filtrado.sort_values(
        by=["Expectancy", "p_win", "p_loss"],
        ascending=[False, False, True]
    )

    top5 = df_ranking.head(5).copy()

    # ============================================================
    # FORMATAÇÃO
    # ============================================================

    for col in ["p_win", "p_loss"]:
        df[col] = df[col] * 100

    df["Expectancy_%"] = df["Expectancy"] * 100
    df["Profit_Factor"] = df["Profit_Factor"]

    df_view = df[[
        "Ativo", "Trades", "p_win", "p_loss", "Expectancy_%", "Profit_Factor"
    ]].copy()

    df_view = df_view.rename(columns={
        "p_win": "Prob. +2% (%)",
        "p_loss": "Prob. -1% (%)",
        "Expectancy_%": "Expectativa semanal (%)",
        "Profit_Factor": "Profit Factor"
    })

    top5_view = df_ranking.head(5).copy()

    for col in ["p_win", "p_loss"]:
        top5_view[col] = top5_view[col] * 100

    top5_view["Expectancy_%"] = top5_view["Expectancy"] * 100

    top5_view = top5_view[[
        "Ativo", "Trades", "p_win", "p_loss", "Expectancy_%", "Profit_Factor"
    ]]

    top5_view = top5_view.rename(columns={
        "p_win": "Prob. +2% (%)",
        "p_loss": "Prob. -1% (%)",
        "Expectancy_%": "Expectativa semanal (%)",
        "Profit_Factor": "Profit Factor"
    })

    aba1, aba2 = st.tabs([
        "Base completa dos ativos",
        "Ranking estatístico – Top 5"
    ])

    with aba1:
        st.subheader("Resultado completo do estudo estatístico")
        st.dataframe(
            df_view.sort_values(
                by="Expectativa semanal (%)",
                ascending=False
            ),
            use_container_width=True
        )

    with aba2:
        st.subheader("Top 5 – maior expectativa estatística semanal (+2% / -1%)")
        st.dataframe(top5_view, use_container_width=True)

        st.markdown("""
Critério de classificação:

1. Maior expectativa estatística semanal  
2. Maior probabilidade de atingir +2%  
3. Menor probabilidade de atingir −1%  

Filtro obrigatório:
Probabilidade de ganho maior que a probabilidade de perda.
""")
