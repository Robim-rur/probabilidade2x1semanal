import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Ranking estatístico – maior probabilidade de gain (≈ 2 semanas)")

st.caption(
    "Ranking estatístico de curto prazo priorizando a MAIOR probabilidade de gain. "
    "Somente combinações onde a probabilidade de gain é maior que a de loss são consideradas."
)

# ============================================================
# LISTA DOS ATIVOS
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
# PARÂMETROS
# ============================================================

MAX_DAYS = 10
YEARS_BACK = 10
MIN_TRADES = 60

GAIN_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
LOSS_GRID = [0.005, 0.01, 0.015, 0.02, 0.03]

# ============================================================
# FUNÇÕES
# ============================================================

@st.cache_data(show_spinner=False)
def baixar_dados(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)


def simular(df, gain, loss):

    wins = 0
    losses = 0
    trades = 0

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    for i in range(len(df) - MAX_DAYS - 1):

        entry = closes[i]
        alvo = entry * (1 + gain)
        stop = entry * (1 - loss)

        resultado = None

        for j in range(1, MAX_DAYS + 1):

            idx = i + j

            if highs[idx] >= alvo and lows[idx] <= stop:
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
    expectancy = p_win * gain - p_loss * loss

    return p_win, p_loss, expectancy, trades


# ============================================================
# EXECUÇÃO
# ============================================================

if st.button("Gerar ranking (priorizando probabilidade de gain)"):

    end = datetime.today()
    start = end - timedelta(days=YEARS_BACK * 365)

    dados = {}

    barra = st.progress(0.0)

    for i, ticker in enumerate(ativos_scan):

        try:
            df = baixar_dados(ticker, start, end)

            if df is None or len(df) < 150:
                continue

            df = df[["Open", "High", "Low", "Close"]].dropna()

            dados[ticker] = df

        except:
            pass

        barra.progress((i + 1) / len(ativos_scan))

    resultados = []

    for ticker, df in dados.items():

        melhor = None

        for gain in GAIN_GRID:
            for loss in LOSS_GRID:

                if gain <= loss:
                    continue

                r = simular(df, gain, loss)

                if r is None:
                    continue

                p_win, p_loss, expectancy, trades = r

                if trades < MIN_TRADES:
                    continue

                # REGRA NOVA
                if p_win <= p_loss:
                    continue

                if melhor is None:
                    melhor = {
                        "Ativo": ticker.replace(".SA", ""),
                        "Gain (%)": gain * 100,
                        "Loss (%)": loss * 100,
                        "Prob gain em 2 semanas (%)": p_win * 100,
                        "Prob loss em 2 semanas (%)": p_loss * 100,
                        "expectancy": expectancy,
                        "Trades": trades
                    }
                else:
                    if p_win > melhor["Prob gain em 2 semanas (%)"] / 100:
                        melhor = {
                            "Ativo": ticker.replace(".SA", ""),
                            "Gain (%)": gain * 100,
                            "Loss (%)": loss * 100,
                            "Prob gain em 2 semanas (%)": p_win * 100,
                            "Prob loss em 2 semanas (%)": p_loss * 100,
                            "expectancy": expectancy,
                            "Trades": trades
                        }
                    elif np.isclose(p_win, melhor["Prob gain em 2 semanas (%)"] / 100) and expectancy > melhor["expectancy"]:
                        melhor = {
                            "Ativo": ticker.replace(".SA", ""),
                            "Gain (%)": gain * 100,
                            "Loss (%)": loss * 100,
                            "Prob gain em 2 semanas (%)": p_win * 100,
                            "Prob loss em 2 semanas (%)": p_loss * 100,
                            "expectancy": expectancy,
                            "Trades": trades
                        }

        if melhor is not None:
            resultados.append(melhor)

    if len(resultados) == 0:
        st.error("Nenhum ativo apresentou probabilidade de gain maior que a de loss.")
        st.stop()

    df = pd.DataFrame(resultados)

    df = df.sort_values(
        by=["Prob gain em 2 semanas (%)", "Prob loss em 2 semanas (%)", "expectancy"],
        ascending=[False, True, False]
    )

    melhor_ativo = df.iloc[0]

    st.success(
        f'''Melhor ativo hoje para abrir operação: {melhor_ativo["Ativo"]}

Gain: {melhor_ativo["Gain (%)"]:.2f}%
Loss: {melhor_ativo["Loss (%)"]:.2f}%

Em até aproximadamente 2 semanas:
Probabilidade de gain: {melhor_ativo["Prob gain em 2 semanas (%)"]:.1f}%
Probabilidade de loss: {melhor_ativo["Prob loss em 2 semanas (%)"]:.1f}%'''
    )

    st.subheader("Ranking – maior probabilidade de gain (do melhor para o pior)")

    df_exibicao = df.drop(columns=["expectancy"])

    st.dataframe(df_exibicao, use_container_width=True)

    st.caption(
        "Ranking estatístico priorizando probabilidade de gain. "
        "Somente combinações com P(gain) > P(loss). "
        "Janela máxima de 10 pregões (~2 semanas)."
    )
