import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Calculadora de Opções", layout="centered")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# === Funções ===
def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T == 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo_option_price(S, K, T, r, sigma, tipo='europeia', q=0.0, num_sim=10000):
    np.random.seed(42)
    if tipo == 'europeia':
        ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(num_sim))
        payoff = np.maximum(ST - K, 0)
    elif tipo == 'asiatica':
        steps = max(1, int(T * 252))
        dt = T / steps
        payoff = []
        for _ in range(num_sim):
            prices = [S]
            for _ in range(steps):
                drift = (r - q - 0.5 * sigma ** 2) * dt
                shock = sigma * np.sqrt(dt) * np.random.randn()
                prices.append(prices[-1] * np.exp(drift + shock))
            ST_avg = np.mean(prices)
            payoff.append(max(ST_avg - K, 0))
    return np.exp(-r * T) * np.mean(payoff)

def binomial_american_call(S, K, T, r, sigma, q=0.0, steps=1000):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    payoff = np.maximum(ST - K, 0)
    for i in range(steps - 1, -1, -1):
        ST = ST[:-1] / u
        payoff = np.exp(-r * dt) * (p * payoff[1:] + (1 - p) * payoff[:-1])
        payoff = np.maximum(payoff, ST - K)
    return payoff[0]

# === Interface visual por etapas ===
if "page" not in st.session_state:
    st.session_state.page = 0

steps = [
    "Informe o preço atual do ativo (R$)",
    "Informe o valor de exercício (strike) da opção (R$)",
    "Informe o tempo até o vencimento (em anos)",
    "Informe a taxa de juros anual (% a.a.)",
    "Informe o dividend yield anual (% a.a.)",
    "Informe a volatilidade anual da ação (% a.a.)",
    "Escolha o tipo de opção",
]

# Inputs
if st.session_state.page == 0:
    S = st.number_input(steps[0], min_value=0.01, format="%.2f", key="S")
elif st.session_state.page == 1:
    K = st.number_input(steps[1], min_value=0.01, format="%.2f", key="K")
elif st.session_state.page == 2:
    T = st.number_input(steps[2], min_value=0.01, format="%.4f", key="T")
elif st.session_state.page == 3:
    r = st.number_input(steps[3], min_value=0.0, format="%.2f", key="r") / 100
elif st.session_state.page == 4:
    q = st.number_input(steps[4], min_value=0.0, format="%.2f", key="q") / 100
elif st.session_state.page == 5:
    sigma = st.number_input(steps[5], min_value=0.01, format="%.2f", key="sigma") / 100
elif st.session_state.page == 6:
    tipo = st.selectbox(steps[6], ["Europeia", "Americana", "Asiática"], key="tipo")

# Botões
col1, col2 = st.columns([1, 5])
with col1:
    if st.session_state.page > 0:
        if st.button("Voltar"):
            st.session_state.page -= 1
with col2:
    if st.session_state.page < len(steps) - 1:
        if st.button("Próximo"):
            st.session_state.page += 1
    else:
        if st.button("Calcular Preço"):
            S = st.session_state.S
            K = st.session_state.K
            T = st.session_state.T
            r = st.session_state.r
            q = st.session_state.q
            sigma = st.session_state.sigma
            tipo = st.session_state.tipo

            if tipo == "Europeia":
                preco = bs_call_price(S, K, T, r, sigma, q)
            elif tipo == "Asiática":
                preco = monte_carlo_option_price(S, K, T, r, sigma, tipo="asiatica", q=q)
            elif tipo == "Americana":
                preco = binomial_american_call(S, K, T, r, sigma, q=q)

            st.markdown(f"<div class='big-font'>💰 Preço estimado da opção {tipo.lower()}: R$ {preco:,.4f}</div>", unsafe_allow_html=True)
