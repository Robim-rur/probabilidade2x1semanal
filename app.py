import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Scanner Estatístico – Probabilidade de Alta Semanal (+2% / -1%)")

# =========================================================
# PARÂMETROS FIXOS DO MODELO
# =========================================================
TARGET = 0.02   # +2%
STOP   = -0.01  # -1%
HORIZON = 5     # 5 pregões

# =========================================================
# LISTA DE ATIVOS (exemplo – você pode expandir)
# =========================================================
default_tickers = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "SUZB3.SA"
]

tickers = st.multiselect(
    "Ativos para análise",
    options=default_tickers,
    default=default_tickers
)

anos = st.slider("Anos de histórico para o estudo", 5, 12, 10)

# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def estudar_ativo(ticker, anos):

    fim = datetime.today()
    inicio = fim - timedelta(days=365 * anos)

    df = yf.download(
        ticker,
        start=inicio.strftime("%Y-%m-%d"),
        end=fim.strftime("%Y-%m-%d"),
        progress=False
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

        bateu_alvo = False
        bateu_stop = False
        ordem = None

        max_retorno = -999
        max_drawdown = 999

        for j in range(1, HORIZON + 1):

            h = highs[i + j]
            l = lows[i + j]

            r_max = (h / entrada) - 1
            r_min = (l / entrada) - 1

            max_retorno = max(max_retorno, r_max)
            max_drawdown = min(max_drawdown, r_min)

            if (h >= alvo) and (l <= stop):
                ordem = "ambos"
                break

            if h >= alvo:
                bateu_alvo = True
                ordem = "alvo"
                break

            if l <= stop:
                bateu_stop = True
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
    sucessos = r["sucesso"].sum()
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

    with st.spinner("Processando..."):

        for t in tickers:
            r = estudar_ativo(t, anos)
            if r is not None:
                resultados_finais.append(r)

    if len(resultados_finais) == 0:
        st.warning("Nenhum ativo retornou dados suficientes.")
    else:

        df_final = pd.DataFrame(resultados_finais)

        df_final = df_final.sort_values(
            by="Probabilidade_alvo_2pct",
            ascending=False
        )

        st.subheader("Resultado – Estatística do evento (+2% antes de -1% em até 5 pregões)")

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

        st.dataframe(
            tabela,
            use_container_width=True
        )

        st.markdown("### Interpretação das colunas")
        st.markdown("""
- **Amostras** → quantidade de janelas históricas testadas.
- **Probabilidade de bater +2% (%)** → percentual de vezes em que o preço atingiu +2% antes de -1% em até 5 pregões.
- **Máx. retorno médio no período (%)** → média do melhor retorno observado dentro da janela de 5 dias.
- **Drawdown médio no período (%)** → média da pior variação negativa dentro da janela.
- **Payoff médio** → relação entre retorno potencial médio e risco médio.
- **Preço atual** → último fechamento.
        """)

        csv = tabela.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar tabela em CSV",
            csv,
            "estatistica_2pct_1pct.csv",
            "text/csv"
        )
