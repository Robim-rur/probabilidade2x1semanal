import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Detector de Regime Operável – Setup Principal e SAR")

# ==========================================================
# LISTA DE ATIVOS
# ==========================================================

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

# ==========================================================
# INDICADORES
# ==========================================================

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def dmi_adx(df, n=14):

    up = df['High'].diff()
    down = -df['Low'].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    plus_di = 100 * plus_dm.rolling(n).mean() / atr
    minus_di = 100 * minus_dm.rolling(n).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(n).mean()

    return plus_di, minus_di, adx

# ==========================================================
# SAR PARABÓLICO (IMPLEMENTAÇÃO CORRIGIDA)
# ==========================================================

def parabolic_sar(df, step=0.02, max_step=0.2):

    high = df['High'].values
    low = df['Low'].values

    sar = np.zeros(len(df))
    trend = 1
    af = step

    ep = high[0]
    sar[0] = low[0]

    for i in range(1, len(df)):

        sar[i] = sar[i-1] + af * (ep - sar[i-1])

        if trend == 1:

            sar[i] = min(sar[i], low[i-1], low[i])

            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)

            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = step

        else:

            sar[i] = max(sar[i], high[i-1], high[i])

            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)

            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step

    return pd.Series(sar, index=df.index)

# ==========================================================
# PROCESSAMENTO
# ==========================================================

@st.cache_data(show_spinner=False)
def processar():

    registros_setup = []
    registros_sar = []

    hoje_setup = []
    hoje_sar = []

    for ticker in ativos_scan:

        try:

            df = yf.download(ticker, period="12y", interval="1d", progress=False)

            if df is None or len(df) < 300:
                continue

            df = df.dropna()

            df['EMA69'] = ema(df['Close'], 69)

            pdi, mdi, adx = dmi_adx(df)

            df['PDI'] = pdi
            df['MDI'] = mdi
            df['ADX'] = adx

            df['SAR'] = parabolic_sar(df)

            weekly = df.resample('W-FRI').last()
            weekly['EMA69_W'] = ema(weekly['Close'], 69)

            df = df.merge(weekly[['EMA69_W']], left_index=True, right_index=True, how='left')
            df['EMA69_W'] = df['EMA69_W'].ffill()

            df['setup_principal'] = (
                (df['Close'] > df['EMA69']) &
                (df['Close'] > df['EMA69_W']) &
                (df['PDI'] > df['MDI'])
            )

            df['setup_sar'] = (
                (df['Close'] > df['SAR']) &
                (df['Close'] > df['EMA69']) &
                (df['Close'] > df['EMA69_W'])
            )

            df['ret_futuro'] = df['Close'].shift(-10) / df['Close'] - 1

            for i in range(len(df) - 10):

                if df['setup_principal'].iloc[i]:
                    registros_setup.append(df['ret_futuro'].iloc[i])

                if df['setup_sar'].iloc[i]:
                    registros_sar.append(df['ret_futuro'].iloc[i])

            if df['setup_principal'].iloc[-1]:
                hoje_setup.append(ticker)

            if df['setup_sar'].iloc[-1]:
                hoje_sar.append(ticker)

        except:
            pass

    return registros_setup, registros_sar, hoje_setup, hoje_sar

with st.spinner("Processando histórico..."):
    reg_setup, reg_sar, hoje_setup, hoje_sar = processar()

def resumo(reg):

    if len(reg) == 0:
        return 0, 0, 0

    s = pd.Series(reg).dropna()

    win = (s > 0).mean() * 100
    ret_medio = s.mean() * 100
    perda_media = s[s < 0].mean() * 100 if (s < 0).any() else 0

    return win, ret_medio, perda_media

win_s, ret_s, dd_s = resumo(reg_setup)
win_sar, ret_sar, dd_sar = resumo(reg_sar)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Setup principal (seu setup)")
    st.metric("Probabilidade histórica de ganho (10 pregões)", f"{win_s:.2f}%")
    st.metric("Retorno médio", f"{ret_s:.2f}%")
    st.metric("Perda média", f"{dd_s:.2f}%")
    st.metric("Ativos em condição hoje", len(hoje_setup))
    st.dataframe(pd.DataFrame({"Ativos": hoje_setup}), use_container_width=True)

with col2:

    st.subheader("Setup SAR Parabólico")
    st.metric("Probabilidade histórica de ganho (10 pregões)", f"{win_sar:.2f}%")
    st.metric("Retorno médio", f"{ret_sar:.2f}%")
    st.metric("Perda média", f"{dd_sar:.2f}%")
    st.metric("Ativos em condição hoje", len(hoje_sar))
    st.dataframe(pd.DataFrame({"Ativos": hoje_sar}), use_container_width=True)

st.markdown("---")

def regime(win):

    if win >= 55:
        return "REGIME FAVORÁVEL"
    elif win >= 50:
        return "REGIME NEUTRO"
    else:
        return "REGIME DESFAVORÁVEL"

st.subheader("Diagnóstico de regime")

st.write("Setup principal:", regime(win_s))
st.write("Setup SAR:", regime(win_sar))

st.info(
"""
Este app não gera sinal de entrada.
Ele mede se o ambiente favorece o seu setup principal
e o setup com SAR parabólico, usando horizonte de 10 pregões.
"""
)
