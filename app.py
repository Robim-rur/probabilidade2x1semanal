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
# PARÂMETROS FIXOS DO MODELO
# =========================================================

TARGET = 0.02   # +2%
STOP   = -0.01  # -1%
HORIZON = 5     # 5 pregões

# =========================================================
# LISTA FIXA DE ATIVOS (fornecida por você)
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
    dates = df.index

    resultados = []

    for i in range(len(df) - HORIZON - 1):

        entrada = closes[i]

        alvo = entrada * (1 + TARGET)
        stop = entrada * (1 + STOP)

        ordem = None

        max_retorno = -999.0
        max_drawdown = 999.0

        for j in range(1, HORIZON + 1):

            h = highs[i + j]
            l = lows[i + j]

            r_max = (h / entrada) - 1
            r_min = (l / entrada) - 1

            if r_max > max_retorno:
                max_retorno = r_max

            if r_min < max_drawdown:
                max_drawdown = r_min

            if (h >= alvo) and (l <= stop):
                ordem = "ambos"
                break

            if h >= alvo:
                ordem = "alvo"
                break

            if l <= stop:
                ordem = "stop"
                break

        sucesso = 1 if ordem == "alvo" else 0

        resultados.append({
            "data": dates[i],
            "sucesso": sucesso,
            "ordem": ordem,
            "max_retorno": max_retorno,
            "max_drawdown": max_drawdown
        })

    if len(resultados) == 0:
        return None

    r = pd.DataFrame(resultados)

    total = len(r)
    sucessos = int(r["sucesso"].sum())
    taxa = sucessos / total

    retorno_medio_max = r["max_retorno"].mean()
    drawdown_medio = r["max_drawdown"].mean()

    payoff = abs(retorno_medio_max / drawdown_medio) if drawdown_medio != 0 else np.nan

    ult_close = closes[-1]

    return {
        "Ativo": ticker.replace(".SA", ""),
        "Amostras": total,
        "Probabilidade_alvo_2pct": taxa,
        "Retorno_max_médio": retorno_medio_max,
        "Drawdown_médio": drawdown_medio,
        "Payoff_aproximado": payoff,
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

        df_final = pd.DataFrame(resultados_finais)

        df_final = df_final.sort_values(
            by="Probabilidade_alvo_2pct",
            ascending=False
        )

        tabela = df_final.copy()

        tabela["Probabilidade_alvo_2pct"] = tabela["Probabilidade_alvo_2pct"] * 100
        tabela["Retorno_max_médio"] = tabela["Retorno_max_médio"] * 100
        tabela["Drawdown_médio"] = tabela["Drawdown_médio"] * 100

        tabela = tabela.rename(columns={
            "Probabilidade_alvo_2pct": "Probabilidade de bater +2% (%)",
            "Retorno_max_médio": "Máx. retorno médio no período (%)",
            "Drawdown_médio": "Drawdown médio no período (%)",
            "Payoff_aproximado": "Payoff médio",
            "Preço_atual": "Preço atual"
        })

        st.subheader("Resultado – Evento: +2% antes de −1% em até 5 pregões")

        st.dataframe(tabela, use_container_width=True)

        csv = tabela.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Baixar tabela em CSV",
            csv,
            "resultado_prob2x1_universo_fixo.csv",
            "text/csv"
        )
