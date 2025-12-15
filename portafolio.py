import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(page_title="Gestor de Portafolios", layout="wide")

# ==========================================
# 1. DEFINICIÓN DE UNIVERSOS Y BENCHMARKS
# ==========================================
universos = {
    "Regiones": {
        "SPLG": "USA (S&P 500)",
        "EWC": "Canadá",
        "IEUR": "Europa",
        "EEM": "Mercados Emergentes",
        "EWJ": "Japón"
    },
    "Sectores": {
        "XLC": "Comunicación",
        "XLY": "Consumo Discrecional",
        "XLP": "Consumo Básico",
        "XLE": "Energía",
        "XLF": "Financiero",
        "XLV": "Salud",
        "XLI": "Industrial",
        "XLB": "Materiales",
        "XLRE": "Bienes Raíces",
        "XLK": "Tecnología",
        "XLU": "Utilities"
    }
}

# [cite_start]Pesos fijos definidos en el PDF del proyecto [cite: 15]
benchmarks_def = {
    "Regiones": {
        "SPLG": 0.7062, "EWC": 0.0323, "IEUR": 0.1176, "EEM": 0.0902, "EWJ": 0.0537
    },
    "Sectores": {
        "XLC": 0.0999, "XLY": 0.1025, "XLP": 0.0482, "XLE": 0.0295, "XLF": 0.1307,
        "XLV": 0.0958, "XLI": 0.0809, "XLB": 0.0166, "XLRE": 0.0187, "XLK": 0.3535, "XLU": 0.0237
    }
}

# ==========================================
# 2. CARGA DE DATOS (CACHE)
# ==========================================

@st.cache_data(show_spinner=True)
def descargar_todos_los_datos():
    todos_tickers = list(universos["Regiones"].keys()) + list(universos["Sectores"].keys())
    todos_tickers = list(set(todos_tickers))
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    
    try:
        data = yf.download(todos_tickers, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                precios = data['Close']
            except KeyError:
                precios = data
        else:
            precios = data['Close'] if 'Close' in data.columns else data
        return precios
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

with st.spinner("Conectando con Yahoo Finance..."):
    df_precios_totales = descargar_todos_los_datos()

# ==========================================
# 3. FUNCIONES DE CÁLCULO
# ==========================================

def filtrar_datos_universo(precios_totales, tickers_universo):
    cols_validas = [t for t in tickers_universo if t in precios_totales.columns]
    return precios_totales[cols_validas].dropna()

def calcular_metricas(pesos, rendimientos):
    port_returns = rendimientos.dot(pesos)
    mean_ret = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)
    sharpe = (mean_ret - 0.04) / volatility 
    
    downside_returns = port_returns[port_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (mean_ret - 0.04) / downside_std if downside_std > 0 else 0
    
    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    max_drawdown = ((cumulative - peak) / peak).min()
    
    # VaR 95%
    var_95 = np.percentile(port_returns, 5)
    
    # CVaR 95% (Promedio de los rendimientos que son peores que el VaR)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    # Beta
    benchmark_ret = rendimientos.mean(axis=1)
    if volatility > 0 and benchmark_ret.std() > 0:
        beta = np.cov(port_returns, benchmark_ret)[0, 1] / np.var(benchmark_ret)
    else:
        beta = 0
    
    return {
        "Rendimiento Esp.": mean_ret,
        "Volatilidad": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "VaR (95%)": var_95,
        "CVaR (95%)": cvar_95,  
        "Beta": beta
    }

def portafolio_min_varianza(cov_matrix_anual):
    n = len(cov_matrix_anual)
    # Objetivo: Minimizar la volatilidad anualizada
    def objective(weights): 
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_anual, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #limite = min(1 / np.sqrt(n), 0.25)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # Init guess: Pesos iguales
    init_guess = n * [1. / n]
    
    # Usamos tol (tolerancia) más estricta para forzarlo a buscar mejor
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    return result.x

def portafolio_max_sharpe(mu_anual, cov_matrix_anual, rf=0.04):
    n = len(mu_anual)
    # Objetivo: Maximizar Sharpe (Minimizar -Sharpe)
    def objective(weights):
        port_ret = np.dot(weights, mu_anual)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_anual, weights)))
        return -(port_ret - rf) / port_vol 
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #limite = min(1 / np.sqrt(n), 0.25)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init_guess = n * [1. / n]
    
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-10)
    return result.x

def risk_aversion_coefficient(returns, rf=0.04):
    """Calcula el coeficiente de aversión al riesgo λ"""
    mean = returns.mean().mean() * 252
    var = returns.mean(axis=1).var() * 252
    return (mean - rf) / var if var > 0 else 2.5

def implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion):
    """
    Calcula los retornos implícitos de equilibrio: Π = λ Σ w_mkt
    """
    return risk_aversion * cov_matrix @ market_weights

def build_P_Q_from_scores(scores):
    """
    Construye las matrices P y Q desde los scores del usuario
    P: matriz identidad (cada view es sobre un activo)
    Q: vector de views (1% por cada punto de score)
    """
    n = len(scores)
    P = np.eye(n)
    Q = np.array(scores) * 0.01  # 1% por punto de score
    return P, Q

def build_omega(P, cov_matrix, confidences, tau=0.05):
    """
    Construye la matriz de incertidumbre de las views: Ω
    Ω_k = ((1 - c_k) / c_k) * (P_k @ (τ * Σ) @ P_k')
    
    Donde:
    - c_k es la confianza de la view k (entre 0 y 1)
    - Menor confianza → Mayor incertidumbre → Mayor Ω
    """
    Sigma = cov_matrix if isinstance(cov_matrix, np.ndarray) else cov_matrix.values
    Sigma_tau = tau * Sigma
    
    omega_diag = []
    for i, c in enumerate(confidences):
        c = max(c, 1e-4)  # Evitar división por cero
        c = min(c, 0.9999)  # Evitar valores extremos
        p_i = P[i].reshape(1, -1)
        variance = p_i @ Sigma_tau @ p_i.T
        omega_diag.append(((1 - c) / c) * variance[0, 0])
    
    return np.diag(omega_diag)

def black_litterman_returns(cov_matrix, pi, P, Q, omega, tau=0.05):
    """
    Calcula los retornos esperados según Black-Litterman
    
    μ_BL = [Σ_τ^-1 + P' Ω^-1 P]^-1 @ [Σ_τ^-1 π + P' Ω^-1 Q]
    
    Donde Σ_τ = τ * Σ
    """
    try:
        Sigma = cov_matrix if isinstance(cov_matrix, np.ndarray) else cov_matrix.values
        Sigma_tau = tau * Sigma
        
        # Cálculo directo según la fórmula
        inv_Sigma_tau = np.linalg.inv(Sigma_tau)
        inv_Omega = np.linalg.inv(omega)
        
        # Parte izquierda: [Σ_τ^-1 + P' Ω^-1 P]^-1
        precision_posterior = np.linalg.inv(
            inv_Sigma_tau + P.T @ inv_Omega @ P
        )
        
        # Parte derecha: [Σ_τ^-1 π + P' Ω^-1 Q]
        combined_info = inv_Sigma_tau @ pi + P.T @ inv_Omega @ Q
        
        mu_bl = precision_posterior @ combined_info
        return mu_bl
    except np.linalg.LinAlgError:
        st.warning(" Error en inversión de matriz BL, usando retornos de equilibrio")
        return pi
# ==========================================
# 4. INTERFAZ
# ==========================================
if 'step' not in st.session_state:
    st.session_state.step = 'planteamiento'

def reiniciar():
    st.session_state.step = 'planteamiento'

# --- PANTALLA 1: INPUTS ---
if st.session_state.step == 'planteamiento':
    st.title("Planteamiento del Portafolio")
    
    col_univ, col_tau = st.columns([3, 1])
    with col_univ:
        opcion_universo = st.selectbox("Seleccionar Universo:", ["Regiones", "Sectores"])
    with col_tau:
        tau_param = st.number_input("τ (tau)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, 
                                     help="Incertidumbre en retornos de equilibrio (típicamente 0.025-0.05)")
    

    data_dict = universos[opcion_universo]

    if 'datos_usuario' in st.session_state and st.session_state.get('tipo_universo') == opcion_universo:
        df_input = st.session_state['datos_usuario']
    else:
        df_input = pd.DataFrame({
            "Ticker": data_dict.keys(),
            "Descripción": data_dict.values(),
            "View (Score)": 0
        })

    st.info("Asigne scores entre -3 (Bajista) y +3 (Alcista). **La suma debe ser 0**.")
    
    edited_df = st.data_editor(
        df_input,
        column_config={
            "View (Score)": st.column_config.NumberColumn(
                "View", min_value=-3, max_value=3, step=1, format="%d"
            )
        },
        disabled=["Ticker", "Descripción"],
        hide_index=True,
        use_container_width=True,
        key="editor_views"
    )

    suma = edited_df["View (Score)"].sum()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if suma == 0:
            st.success(f"Balanceado (Suma: {suma})")
            disabled_btn = False
        else:
            st.error(f"Desbalanceado (Suma: {suma}). Debe ser 0.")
            disabled_btn = True
            
    with col2:
        if st.button("Generar Portafolios", disabled=disabled_btn, type="primary"):
            st.session_state['datos_usuario'] = edited_df
            st.session_state['tipo_universo'] = opcion_universo
            st.session_state.step = 'metricas'
            st.rerun()

# --- PANTALLA 2: RESULTADOS ---
elif st.session_state.step == 'metricas':
    st.title("Análisis de Portafolios")
    if st.button("⬅Volver al planteamiento"):
        reiniciar()
        st.rerun()
    st.divider()

    df_user = st.session_state['datos_usuario']
    universo_actual = st.session_state['tipo_universo']
    tickers_solicitados = df_user["Ticker"].tolist()
    scores = df_user["View (Score)"].tolist()
    tau = st.session_state.get('tau_param', 0.05)

    # 1. FILTRAR DATOS
    prices = filtrar_datos_universo(df_precios_totales, tickers_solicitados)
    
    if prices.empty:
        st.error("No se encontraron datos.")
    else:
        returns = prices.pct_change().dropna()
        
        # --- PRE-CÁLCULO ANUALIZADO (CLAVE PARA QUE EL OPTIMIZADOR FUNCIONE) ---
        mu_anual = returns.mean() * 252
        cov_anual = returns.cov() * 252  # <--- Esto arregla el problema de Min Varianza
        # CALCULOS DEL BENCHMARK 
        dict_pesos_bench = benchmarks_def[universo_actual]
        lista_pesos_bench = [dict_pesos_bench.get(t, 0.0) for t in tickers_solicitados]
        w_benchmark = np.array(lista_pesos_bench)
        # 2. CÁLCULO DE ESTRATEGIAS ACTIVAS
        try:
            n_assets = len(tickers_solicitados)
            w_eq = np.array([1/n_assets] * n_assets) 
            
            # Pasamos las matrices YA anualizadas a los optimizadores
            w_mv = portafolio_min_varianza(cov_anual) 
            w_ms = portafolio_max_sharpe(mu_anual, cov_anual) 
            lambda_ = risk_aversion_coefficient(returns)
            pi = implied_equilibrium_returns(cov_anual, w_benchmark, lambda_)
            P, Q = build_P_Q_from_scores(scores)
            
            confidences = (np.abs(np.array(scores)) / 3) * 0.8 + 0.2
            omega = build_omega(P, cov_anual, confidences, tau=tau)
            
            mu_bl = black_litterman_returns(cov_anual, pi, P, Q, omega)
            w_bl = portafolio_min_varianza(cov_anual)

            
            # Agrupamos las 5 estrategias para iterar
            strategies = {
                "Equitativo": w_eq,
                "Min. Varianza": w_mv,
                "Max. Sharpe": w_ms,
                "Black-Litterman": w_bl,
                "Benchmark": w_benchmark  # Agregado como 5ta estrategia
            }
            
            # 3. VISUALIZACIÓN DE ESTRATEGIAS (5 COLUMNAS)
            st.subheader("Asset Allocation")
            
            # Creamos 5 columnas para los pays
            cols = st.columns(5)
            
            results_metrics = {}
            
            for i, (name, weights) in enumerate(strategies.items()):
                # Calcular métricas para todas (incluyendo Benchmark)
                results_metrics[name] = calcular_metricas(weights, returns)
                
                df_pie = pd.DataFrame({"Asset": tickers_solicitados, "Weight": weights})
                df_pie = df_pie[df_pie["Weight"] > 0.005] 
                
                fig = px.pie(df_pie, values="Weight", names="Asset", title=name, hole=0.4)
                fig.update_layout(showlegend=False, height=200, margin=dict(t=30, b=0, l=10, r=10))
                
                # Asignar a la columna correspondiente
                if i < 5:
                    cols[i].plotly_chart(fig, use_container_width=True)

            # TABLA PRINCIPAL DE MÉTRICAS (Ahora incluye Benchmark)
            st.subheader("Comparativa de Estrategias vs Benchmark")
            df_res = pd.DataFrame(results_metrics)
            
            filas_porcentaje = ["Rendimiento Esp.", "Volatilidad", "Max Drawdown", "VaR (95%)"]
            filas_numero = ["Sharpe Ratio", "Sortino Ratio", "Beta"]

            # Destacamos el Benchmark visualmente al final o con estilo
            st.dataframe(df_res.style
                .format("{:.2%}", subset=pd.IndexSlice[filas_porcentaje, :])  # Formato %
                .format("{:.2f}", subset=pd.IndexSlice[filas_numero, :]),     # Formato número (1.25)
                use_container_width=True
            )
            with st.expander(" Detalles Black-Litterman"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Coef. Aversión al Riesgo (lamda))", f"{lambda_:.2f}")
                    st.metric("Parámetro de Incertidumbre (Tau)", f"{tau:.3f}")
                    
                with col2:
                    st.write("**Retornos de Equilibrio (Π):**")
                    df_pi = pd.DataFrame({
                        "Asset": tickers_solicitados, 
                        "Π (%)": pi * 100
                    })
                    st.dataframe(df_pi.style.format({"Π (%)": "{:.2f}"}), hide_index=True)
                
                with col3:
                    st.write("**Retornos Black-Litterman (μ_BL):**")
                    df_mu_bl = pd.DataFrame({
                        "Asset": tickers_solicitados, 
                        "μ_BL (%)": mu_bl * 100
                    })
                    st.dataframe(df_mu_bl.style.format({"μ_BL (%)": "{:.2f}"}), hide_index=True)
                
                st.divider()

                # Mostrar Views y Confianzas
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Views del Inversionista:**")
                    df_views = pd.DataFrame({
                        "Asset": tickers_solicitados,
                        "Score": scores,
                        "Q (%)": Q * 100,
                        "Confianza": confidences
                    })
                    st.dataframe(df_views.style.format({
                        "Score": "{:.0f}",
                        "Q (%)": "{:.2f}",
                        "Confianza": "{:.2f}"
                    }), hide_index=True)
                
                with col2:
                    st.write("**Matriz de Incertidumbre (Ω diagonal):**")
                    omega_diag = np.diag(omega)
                    df_omega = pd.DataFrame({
                        "Asset": tickers_solicitados,
                        "Ω": omega_diag,
                        "Interpretación": ["Alta" if o > np.median(omega_diag) else "Baja" 
                                          for o in omega_diag]
                    })
                    st.dataframe(df_omega.style.format({"Ω": "{:.6f}"}), hide_index=True)
                    st.caption("Ω alta = Mayor incertidumbre en la view → Menor peso en el ajuste")

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
            import traceback
            st.code(traceback.format_exc())

        #except Exception as e:
         #   st.error(f"Error en el cálculo matemático: {e}")