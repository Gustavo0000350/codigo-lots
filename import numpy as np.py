import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Funções para os cálculos

# Tração Estática
def K(P, D):
    return 57000 * (1.97 - P / D)

def TE(Pe, N, D, P):
    k_value = K(P, D)
    return k_value * (Pe / (N * D))

# Tração Disponível
def TD(Pe, Nh, V, Ph, P0):
    return  Pe * (Nh / V) * (Ph / P0)

# Razão de Avanço
def J(V, N, D):
    return V / (N * D)

# Rotação do Motor
def N_motor(TD, V, Pe):
    return TD * V / Pe

# Configuração dos parâmetros gerais
Pe = 100000   # Potência disponível no eixo motor (em Watts)
N = 1000      # Rotação inicial (em RPM)
Nh = 0.85     # Eficiência da hélice
V = np.linspace(10, 200, 20)  # Velocidade do voo (em m/s)
S = 16.2        # Área da asa (m²)
Cd0 = 0.02      # Coeficiente de arrasto parasita
K = 0.045       # Fator de eficiência induzida
e0 = 0.8        # Eficiência da asa
AR = 7.5        # Alongamento da asa
Ph = 1.0        # Densidade do ar na altitude (kg/m³)
P0 = 1.225      # Densidade do ar ao nível do mar (kg/m³)
Pd0 = 200000    # Potência do motor (W)
g = 9.81                # Aceleração gravitacional (m/s²)
p = 1.225               # Densidade do ar ao nível do mar (kg/m³)
Clmax = 1.5             # Coeficiente máximo de sustentação
T = 15000               # Tração do motor (N)
mu = 0.03               # Coeficiente de atrito
e = 0.8                 # Eficiência da asa
AR = 7.5                # Alongamento da asa
b = 10                  # Envergadura da asa (m)
h = 1.5                 # Altura da asa ao solo (m)
S = 16.2                # Área da asa (m²)
W_range = np.linspace(5000, 15000, 500)  # Intervalo de peso (N)
Vmax = 200              # Velocidade máxima (m/s)
Nmax_range = np.linspace(1, 5, 500)  # Intervalo de cargas máximas (G)
h = np.linspace(0, 10000, 500)       # Altura (m)
Wt = 12000              # Peso total inicial (N)
Wvazio = 7000           # Peso vazio da aeronave (N)
Cu0 = 1000              # Carga útil inicial (kg)
m = 0.1                 # Taxa de redução da carga útil por altura (kg/m)
Cd0 = 0.02  # Coeficiente de arrasto parasita

# Configuração para diferentes hélices
helices = [
    {"nome": "Hélice 1", "P": 2.5, "D": 1.5},
    {"nome": "Hélice 2", "P": 3.0, "D": 1.7},
    {"nome": "Hélice 3", "P": 2.0, "D": 1.3},
]

# Inicializar armazenamento de resultados
resultados = {}

for helice in helices:
    nome = helice["nome"]
    P = helice["P"]
    D = helice["D"]
    
    # Cálculos para a hélice
    TE_values = TE(Pe, N, D, P)
    TD_values = TD(Pe, Nh, V, Ph, P0)
    J_values = J(V, N, D)
    N_values = N_motor(TD_values, V, Pe)
    
    # Armazenar resultados
    resultados[nome] = {
        "Velocidade (V) [m/s]": V,
        "Tração Disponível (TD) [N]": TD_values,
        "Razão de Avanço (J)": J_values,
        "Rotação do Motor (N)": N_values,
        "Tração Estática (TE)": TE_values,
    }

    # Criar DataFrame para a tabela
    df = pd.DataFrame({
        "Velocidade (V) [m/s]": V,
        "Tração Disponível (TD) [N]": TD_values
    })
    print(f"\nTabela para {nome}:")
    print(df)

# Gerar gráficos para cada hélice
for helice in helices:
    nome = helice["nome"]
    P = helice["P"]
    D = helice["D"]
    
    # Recuperar dados
    V = resultados[nome]["Velocidade (V) [m/s]"]
    TD_values = resultados[nome]["Tração Disponível (TD) [N]"]
    J_values = resultados[nome]["Razão de Avanço (J)"]
    N_values = resultados[nome]["Rotação do Motor (N)"]

    # Gráfico de N por J
    plt.figure(figsize=(10, 5))
    plt.plot(J_values, N_values, label=f"N por J - {nome}")
    plt.xlabel("Razão de Avanço (J)")
    plt.ylabel("Rotação do Motor (N)")
    plt.title(f"Gráfico de N por J para {nome}")
    plt.grid()
    plt.legend()
    plt.show()

    # Gráfico de V por TD
    plt.figure(figsize=(10, 5))
    plt.plot(V, TD_values, label=f"V por TD - {nome}")
    plt.xlabel("Velocidade do Voo (V) [m/s]")
    plt.ylabel("Tração Disponível (TD) [N]")
    plt.title(f"Gráfico de V por TD para {nome}")
    plt.grid()
    plt.legend()
    plt.show()

# Funções auxiliares
def Cl(W, p, V, S):
    return 2 * Wt / (p * V**2 * S)

def Cd(Cd0, Cl, K):
    return Cd0 + K * Cl**2

def Tr(Wt, Cl, Cd):
    return W / (Cl / Cd)

def Tdh(Pd0, Nh, V, Ph, P0):
    return (Pd0 * Nh / V) * (Ph / P0)

def Pd(Tdh, V):
    return Tdh * V

def Prh(Wt, Cd, Ph, S, Cl):
    return np.sqrt(2 * Wt * Cd**2 / (Ph * S * Cl**3))

# Cálculos
Cl_values = Cl(Wt, Ph, V=, S)
Cd_values = Cd(Cd0, Cl_values, K)
Tr_values = Tr(Wt=, Cl_values, Cd_values)
Tdh_values = Tdh(Pd0, Nh, V, Ph, P0)
Pd_values = Pd(Tdh_values, V)
Prh_values = Prh(Wt, Cd_values, Ph, S, Cl_values)

# Gráficos
plt.figure(figsize=(10, 5))
plt.plot(V, Tr_values, label="Tração Requerida (Tr)")
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Tração Requerida (Tr) [N]")
plt.title("Gráfico de Tração Requerida (Tr) por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(V, Tdh_values, label="Tração Disponível (Tdh)")
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Tração Disponível (Tdh) [N]")
plt.title("Gráfico de Tração Disponível (Tdh) por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

# Comparação entre Tr e Td
plt.figure(figsize=(10, 5))
plt.plot(V, Tr_values, label="Tração Requerida (Tr)")
plt.plot(V, Tdh_values, label="Tração Disponível (Tdh)", linestyle='--')
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Tração (N)")
plt.title("Comparação de Tr e Td por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

# Potência Disponível
plt.figure(figsize=(10, 5))
plt.plot(V, Pd_values, label="Potência Disponível (Pd)")
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Potência Disponível (Pd) [W]")
plt.title("Gráfico de Potência Disponível (Pd) por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

# Potência Requerida em Altitude
plt.figure(figsize=(10, 5))
plt.plot(V, Prh_values, label="Potência Requerida em Altitude (Prh)")
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Potência Requerida (Prh) [W]")
plt.title("Gráfico de Potência Requerida em Altitude (Prh) por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()
# Funções auxiliares
def Cl(Wt, p, V, S):
    return 2 * Wt / (p * V**2 * S)

def Cd(Cd0, Cl, K):
    return Cd0 + K * Cl**2

def Tr(Wt, Cl, Cd):
    return Wt / (Cl / Cd)

def Tdh(Pd0, Nh, V, Ph, P0):
    return (Pd0 * Nh / V) * (Ph / P0)

def Pd(Tdh, V):
    return Tdh * V

def Prh(Wt, Cd, Ph, S, Cl):
    return np.sqrt(2 * Wt * Cd**2 / (Ph * S * Cl**3))

# Cálculos
Cl_values = Cl(Wt, Ph, V, S)
Cd_values = Cd(Cd0, Cl_values, K)
Tr_values = Tr(Wt, Cl_values, Cd_values)
Tdh_values = Tdh(Pd0, Nh, V, Ph, P0)
Pd_values = Pd(Tdh_values, V)
Prh_values = Prh(W, Cd_values, Ph, S, Cl_values)

# Subida da nave
Fpa_subida = Tr_values + Wt * np.sin(theta)  # Força paralela na subida
Fpe_subida = Wt * np.cos(theta)             # Força perpendicular na subida

# Descida da nave
alpha = np.radians(5)                      # Ângulo de descida (graus convertido para radianos)
Dpa_descida = Wt * np.sin(alpha)            # Força paralela na descida
Dpe_descida = Wt* np.cos(alpha)            # Força perpendicular na descida

# Ângulo de equilíbrio durante a descida
LD_ratio = Cl_values / Cd_values           # Razão de Sustentação/Arrasto
tg_alpha = 1 / LD_ratio                    # Tangente do ângulo de equilíbrio
alpha_equilibrio = np.degrees(np.arctan(tg_alpha))  # Ângulo de equilíbrio em graus

# Gráfico: Comparação entre Prh e Pd
plt.figure(figsize=(10, 5))
plt.plot(V, Prh_values, label="Potência Requerida (Prh)", linestyle='-', color='blue')
plt.plot(V, Pd_values, label="Potência Disponível (Pd)", linestyle='--', color='red')
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Potência (W)")
plt.title("Comparação de Potência Requerida (Prh) e Potência Disponível (Pd) por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

# Gráfico: Forças na subida
plt.figure(figsize=(10, 5))
plt.plot(V, Fpa_subida, label="Força Paralela na Subida (Fpa)", linestyle='-', color='green')
plt.plot(V, Fpe_subida, label="Força Perpendicular na Subida (Fpe)", linestyle='--', color='orange')
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Força (N)")
plt.title("Forças na Subida da Nave")
plt.grid()
plt.legend()
plt.show()

# Gráfico: Forças na descida
plt.figure(figsize=(10, 5))
plt.plot(V, Dpa_descida, label="Força Paralela na Descida (Dpa)", linestyle='-', color='purple')
plt.plot(V, Dpe_descida, label="Força Perpendicular na Descida (Dpe)", linestyle='--', color='brown')
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Força (N)")
plt.title("Forças na Descida da Nave")
plt.grid()
plt.legend()
plt.show()

# Gráfico: Ângulo de equilíbrio durante a descida
plt.figure(figsize=(10, 5))
plt.plot(V, alpha_equilibrio, label="Ângulo de Equilíbrio (α)", linestyle='-', color='teal')
plt.xlabel("Velocidade (V) [m/s]")
plt.ylabel("Ângulo de Equilíbrio (α) [graus]")
plt.title("Ângulo de Equilíbrio durante a Descida por Velocidade (V)")
plt.grid()
plt.legend()
plt.show()

# Funções auxiliares
def AR_calc(b, S):
    return b**2 / S

def Ø(h, b):
    return (16 * h / b)**2 / (1 + (16 * h / b)**2)

def Cllo(e, AR, mu, Ø):
    return np.pi * e * AR * mu / (2 * Ø)

def Vestol(Wt, p, S, Clmax):
    return np.sqrt((2 * Wt) / (p * S * Clmax))

def Vto(Vestol):
    return 1.2 * Vestol

def Slo(Wt, g, p, S, Clmax, T, D, L, mu):
    return 1.44 * Wt**2 / (g * p * S * Clmax * (T - (D + mu * (Wt - L))))

# Cálculos
AR_value = AR_calc(b, S)
Ø_value = Ø(h, b)
Cllo_value = Cllo(e, AR_value, mu, Ø_value)
Vestol_values = Vestol(W_range, p, S, Clmax)
Vto_values = Vto(Vestol_values)

# Força de sustentação (L) e arrasto (D)
L_values = (0.5 * p * Vto_values**2 * S * Clmax)
Cl_values = 2 * W_range / (p * Vto_values**2 * S)
Cd_values = Cd0 + K * Cl_values**2
D_values = 0.5 * p * Vto_values**2 * S * Cd_values

# Comprimento de pista (Slo)
Slo_values = Slo(W_range, g, p, S, Clmax, T, D_values, L_values, mu)

# Tabela de W por Slo
table = pd.DataFrame({"Peso (W) [N]": W_range, "Comprimento da Pista (Slo) [m]": Slo_values})

# Gráfico de W por Slo
plt.figure(figsize=(10, 5))
plt.plot(W_range, Slo_values, label="Comprimento da Pista (Slo)", color="blue")
plt.xlabel("Peso (W) [N]")
plt.ylabel("Comprimento da Pista (Slo) [m]")
plt.title("Gráfico de Peso (W) por Comprimento da Pista (Slo)")
plt.grid()
plt.legend()
plt.show()

# Funções auxiliares
def Ca(L, W):
    return L / Wt

def Camax(p, V, S, Clmax, Wt):
    return p * V**2 * S * Clmax / (2 * Wt)

def Vma(Wt, Nmax, p, S, Clmax):
    return np.sqrt((2 * Wt * Nmax) / (p * S * Clmax))

def Vcru(Vmax):
    return 0.9 * Vmax

def Vd(Vmax):
    return 1.25 * Vmax

def Fr(L, Wt):
    return np.sqrt(L**2 - Wt**2)

def R(V, g, n):
    return V**2 / (g * np.sqrt(n**2 - 1))

def Vrmin(K, Wt, S, p, T_W):
    return np.sqrt(4 * K * (Wt / S) / p * (T_W))

def Rmin(K, Wt, S, p, g, T_W, Cd0):
    return 4 * K * (Wt / S) / (p * g * (T_W)) * np.sqrt(1 - 4 * K * Cd0 / (T_W)**2)

def Tl0(Vlo, a):
    return Vlo / a

def Th(h, R, Clmax):
    return h / (R * Clmax)

def Tcru(Scru, Vcru):
    return Scru / Vcru

def Td(h, RD):
    return h / RD

def Tl(Vestol, a):
    return -Vestol / a

def Cu(Wt, Wvazio, g, m, h):
    return (Wt - Wvazio) / g - m * h

def Tdh(Td0, p, p0):
    return Td0 * (p / p0)

# Cálculos
Vma_values = Vma(W, Nmax_range, p, S, Clmax)
Vcru_value = Vcru(Vmax)
Vd_value = Vd(Vmax)

L = 0.5 * p * Vmax**2 * S * Clmax
Fr_values = Fr(L, W)
R_values = R(Vma_values, g, Nmax_range)

Cu_values = Cu(Wt, Wvazio, g, m, h)

# Tabelas e gráficos
# Tabela: Vma por Nmax
table_vma = pd.DataFrame({"Nmax": Nmax_range, "Velocidade de Manobra (Vma) [m/s]": Vma_values})

# Gráfico: Vma por Nmax
plt.figure(figsize=(10, 5))
plt.plot(Nmax_range, Vma_values, label="Vma por Nmax", color="blue")
plt.xlabel("Carga Máxima (Nmax)")
plt.ylabel("Velocidade de Manobra (Vma) [m/s]")
plt.title("Gráfico de Velocidade de Manobra (Vma) por Carga Máxima (Nmax)")
plt.grid()
plt.legend()
plt.show()

# Gráfico: Cu por h
plt.figure(figsize=(10, 5))
plt.plot(h, Cu_values, label="Carga Útil (Cu) por Altura (h)", color="green")
plt.xlabel("Altura (h) [m]")
plt.ylabel("Carga Útil (Cu) [kg]")
plt.title("Gráfico de Carga Útil (Cu) por Altura (h)")
plt.grid()
plt.legend()
plt.show()

# Tabela: Cu por h
table_cu = pd.DataFrame({"Altura (h) [m]": h, "Carga Útil (Cu) [kg]": Cu_values})

# Exibindo as tabelas
print("Tabela de Velocidade de Manobra (Vma) por Carga Máxima (Nmax):")
print(table_vma.head(10))  # Mostrando as primeiras 10 linhas

print("\nTabela de Carga Útil (Cu) por Altura (h):")
print(table_cu.head(10))  # Mostrando as primeiras 10 linhas