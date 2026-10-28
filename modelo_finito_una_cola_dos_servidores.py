"""Modelo finito de UNA cola y DOS servidores heterogéneos.

Modelo basado en el capítulo proporcionado por el usuario.

Estados:
    0       : sistema vacío.
    1^(1)   : un cliente, atendido por el servidor 1.
    1^(2)   : un cliente, atendido por el servidor 2.
    n>=2    : ambos servidores ocupados y n-2 clientes esperando.

Reglas:
    * Llegadas Poisson de tasa lam > 0.
    * Servidor 1: tasa de servicio mu1 > 0.
    * Servidor 2: tasa de servicio mu2 > 0.
    * Si ambos servidores están libres, una llegada usa el servidor 1.
    * Capacidad total del sistema M. En el estado M, las llegadas se bloquean.

Distribución estacionaria del caso finito:
    rho = lam/(mu1+mu2)
    pi_0      = C K0
    pi_1^(1)  = C K11
    pi_1^(2)  = C K12
    pi_n      = C rho^(n-2),  n=2,...,M

donde
    K0  = mu1*mu2*(2*lam+mu1+mu2) / [lam^2*(lam+mu2)]
    K11 = mu2*(lam+mu1+mu2) / [lam*(lam+mu2)]
    K12 = mu1/(lam+mu2)

Para mitigar errores numéricos, el método principal NO forma rho**(M-2)
ni la suma geométrica de manera directa. Trabaja con log-pesos y logsumexp.
También construye de forma independiente el generador Q para verificar pi Q = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


STATE_0 = "0"
STATE_11 = "1^(1)"
STATE_12 = "1^(2)"


@dataclass(frozen=True)
class Parameters:
    M: int
    lam: float
    mu1: float
    mu2: float

    def __post_init__(self) -> None:
        if not isinstance(self.M, (int, np.integer)) or self.M < 2:
            raise ValueError("M debe ser un entero >= 2.")
        for name, value in (("lambda", self.lam), ("mu1", self.mu1), ("mu2", self.mu2)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} debe ser positivo y finito.")
        if self.mu1 < self.mu2:
            warnings.warn(
                "En el PDF se adopta por convención mu1 >= mu2. "
                "El código funciona con tasas positivas en cualquier orden, pero "
                "la interpretación nominal de servidor 1/2 ya no sigue esa convención.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def muT(self) -> float:
        return self.mu1 + self.mu2

    @property
    def rho(self) -> float:
        return self.lam / self.muT

    @property
    def A(self) -> float:
        return self.lam + self.muT


class FiniteSingleQueueTwoServers:
    """CTMC finita de una cola y dos servidores."""

    def __init__(self, M: int, lam: float, mu1: float, mu2: float):
        self.p = Parameters(M=M, lam=lam, mu1=mu1, mu2=mu2)
        self.states: List[str] = self._build_states()
        self.index: Dict[str, int] = {s: i for i, s in enumerate(self.states)}

    # ------------------------------------------------------------------
    # Estados y transiciones
    # ------------------------------------------------------------------
    def _build_states(self) -> List[str]:
        return [STATE_0, STATE_11, STATE_12] + [str(n) for n in range(2, self.p.M + 1)]

    def total_customers(self, state: str) -> int:
        if state == STATE_0:
            return 0
        if state in (STATE_11, STATE_12):
            return 1
        return int(state)

    def waiting_customers(self, state: str) -> int:
        return max(self.total_customers(state) - 2, 0)

    def server1_busy(self, state: str) -> bool:
        return state == STATE_11 or self.total_customers(state) >= 2

    def server2_busy(self, state: str) -> bool:
        return state == STATE_12 or self.total_customers(state) >= 2

    def transitions(self, state: str, include_blocked_arrival: bool = False):
        """Devuelve lista (evento, destino, tasa).

        Las llegadas bloqueadas se incluyen sólo si include_blocked_arrival=True.
        En el generador Q no se incluyen auto-eventos de bloqueo.
        """
        lam, mu1, mu2, M = self.p.lam, self.p.mu1, self.p.mu2, self.p.M
        out = []

        if state == STATE_0:
            out.append(("llegada", STATE_11, lam))
            return out

        if state == STATE_11:
            out.append(("llegada", "2", lam))
            out.append(("servicio_1", STATE_0, mu1))
            return out

        if state == STATE_12:
            out.append(("llegada", "2", lam))
            out.append(("servicio_2", STATE_0, mu2))
            return out

        n = int(state)

        # Llegada
        if n < M:
            out.append(("llegada", str(n + 1), lam))
        elif include_blocked_arrival:
            out.append(("llegada_bloqueada", state, lam))

        # Salidas. Para n=2, importa qué servidor termina.
        if n == 2:
            out.append(("servicio_1", STATE_12, mu1))
            out.append(("servicio_2", STATE_11, mu2))
        else:
            # n>=3: después de una salida, un cliente de la cola ocupa el servidor liberado.
            out.append(("servicio_1", str(n - 1), mu1))
            out.append(("servicio_2", str(n - 1), mu2))

        return out

    # ------------------------------------------------------------------
    # Generador Q
    # ------------------------------------------------------------------
    def generator(self) -> csr_matrix:
        nstates = len(self.states)
        rows, cols, data = [], [], []

        for state in self.states:
            i = self.index[state]
            rates: Dict[str, float] = {}
            for _, target, rate in self.transitions(state, include_blocked_arrival=False):
                if target == state:
                    continue
                rates[target] = rates.get(target, 0.0) + rate

            total = 0.0
            for target, rate in rates.items():
                rows.append(i)
                cols.append(self.index[target])
                data.append(rate)
                total += rate

            rows.append(i)
            cols.append(i)
            data.append(-total)

        return csr_matrix((data, (rows, cols)), shape=(nstates, nstates), dtype=float)

    # ------------------------------------------------------------------
    # Solución estacionaria estable
    # ------------------------------------------------------------------
    def _log_unnormalized_weights(self) -> np.ndarray:
        """Log-pesos del teorema, sin potencias grandes/pequeñas directas."""
        lam, mu1, mu2 = self.p.lam, self.p.mu1, self.p.mu2
        muT = self.p.muT

        log_lam = np.log(lam)
        log_mu1 = np.log(mu1)
        log_mu2 = np.log(mu2)
        log_rho = log_lam - np.log(muT)

        # K0, K1^(1), K1^(2) en logaritmos.
        logK0 = (
            log_mu1
            + log_mu2
            + np.log(2.0 * lam + mu1 + mu2)
            - 2.0 * log_lam
            - np.log(lam + mu2)
        )
        logK11 = (
            log_mu2
            + np.log(lam + mu1 + mu2)
            - log_lam
            - np.log(lam + mu2)
        )
        logK12 = log_mu1 - np.log(lam + mu2)

        logw = np.empty(len(self.states), dtype=float)
        logw[self.index[STATE_0]] = logK0
        logw[self.index[STATE_11]] = logK11
        logw[self.index[STATE_12]] = logK12

        for n in range(2, self.p.M + 1):
            logw[self.index[str(n)]] = (n - 2) * log_rho

        if not np.all(np.isfinite(logw)):
            raise FloatingPointError("Se generaron log-pesos no finitos.")
        return logw

    def log_stationary_distribution(self) -> pd.Series:
        logw = self._log_unnormalized_weights()
        logpi = logw - logsumexp(logw)
        return pd.Series(logpi, index=self.states, name="log_pi")

    def stationary_distribution(self) -> pd.Series:
        """Distribución estacionaria por fórmula cerrada + logsumexp."""
        logpi = self.log_stationary_distribution().to_numpy(dtype=float)
        pi = np.exp(logpi)
        # Renormalización final para absorber error de redondeo acumulado.
        pi /= pi.sum(dtype=np.float64)
        return pd.Series(pi, index=self.states, name="pi")

    def stationary_via_generator(self) -> pd.Series:
        """Solución independiente de pi Q = 0, sum(pi)=1."""
        Q = self.generator().astype(float)
        n = Q.shape[0]
        scale = max(self.p.A, 1.0)

        Aeq = (Q.T / scale).tolil()
        Aeq[-1, :] = np.ones(n)
        Aeq = csc_matrix(Aeq)

        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.asarray(spsolve(Aeq, b), dtype=float)

        tol = 1e-11
        if np.min(pi) < -tol:
            raise FloatingPointError(
                f"La solución por generador produjo una probabilidad negativa: {pi.min():.3e}"
            )
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum(dtype=np.float64)
        return pd.Series(pi, index=self.states, name="pi_generator")

    def diagnostics(self, pi: pd.Series | None = None) -> dict:
        if pi is None:
            pi = self.stationary_distribution()

        vec = pi.reindex(self.states).to_numpy(dtype=float)
        residual = np.asarray(vec @ self.generator()).ravel()

        result = {
            "num_states": len(self.states),
            "sum_pi": float(vec.sum()),
            "min_pi": float(vec.min()),
            "max_balance_residual": float(np.max(np.abs(residual))),
        }

        # Para tamaños razonables, usamos Q como control independiente.
        if len(self.states) <= 5000:
            pi_q = self.stationary_via_generator().reindex(self.states).to_numpy(dtype=float)
            result["L1_formula_vs_generator"] = float(np.sum(np.abs(vec - pi_q)))
            result["Linf_formula_vs_generator"] = float(np.max(np.abs(vec - pi_q)))

        return result

    # ------------------------------------------------------------------
    # Tablas y medidas de desempeño
    # ------------------------------------------------------------------
    def probability_table(self, pi: pd.Series | None = None) -> pd.DataFrame:
        if pi is None:
            pi = self.stationary_distribution()
        rows = []
        for s in self.states:
            rows.append(
                {
                    "estado": s,
                    "N": self.total_customers(s),
                    "en_espera": self.waiting_customers(s),
                    "servidor_1_ocupado": self.server1_busy(s),
                    "servidor_2_ocupado": self.server2_busy(s),
                    "pi": float(pi.loc[s]),
                }
            )
        return pd.DataFrame(rows)

    def total_distribution(self, pi: pd.Series | None = None) -> pd.Series:
        if pi is None:
            pi = self.stationary_distribution()
        pN = np.zeros(self.p.M + 1, dtype=float)
        for s, prob in pi.items():
            pN[self.total_customers(s)] += float(prob)
        return pd.Series(pN, index=np.arange(self.p.M + 1), name="P(N=n)")

    def metrics(self, pi: pd.Series | None = None) -> dict:
        if pi is None:
            pi = self.stationary_distribution()

        probs = pi.reindex(self.states).to_numpy(dtype=float)
        N = np.array([self.total_customers(s) for s in self.states], dtype=float)
        Qwait = np.maximum(N - 2.0, 0.0)
        b1 = np.array([self.server1_busy(s) for s in self.states], dtype=float)
        b2 = np.array([self.server2_busy(s) for s in self.states], dtype=float)

        p_block = float(pi.loc[str(self.p.M)])
        lam_eff = self.p.lam * (1.0 - p_block)
        EN = float(probs @ N)
        Lq = float(probs @ Qwait)

        # Una llegada espera si ve ambos servidores ocupados pero el sistema no está lleno.
        p_wait_unconditional = float(sum(float(pi.loc[str(n)]) for n in range(2, self.p.M)))
        p_admit = 1.0 - p_block
        p_wait_given_admitted = p_wait_unconditional / p_admit if p_admit > 0 else np.nan

        W = EN / lam_eff if lam_eff > 0 else np.inf
        Wq = Lq / lam_eff if lam_eff > 0 else np.inf

        return {
            "rho": self.p.rho,
            "E[N]": EN,
            "Lq=E[(N-2)+]": Lq,
            "P_bloqueo": p_block,
            "tasa_efectiva": lam_eff,
            "utilizacion_servidor_1": float(probs @ b1),
            "utilizacion_servidor_2": float(probs @ b2),
            "P_espera_no_condicional": p_wait_unconditional,
            "P_espera_dado_admitido": p_wait_given_admitted,
            "W": W,
            "Wq": Wq,
        }

    # ------------------------------------------------------------------
    # Simulación de eventos (Gillespie)
    # ------------------------------------------------------------------
    def simulate(self, T: float, seed: int | None = None, initial: str = STATE_0):
        if not np.isfinite(T) or T <= 0:
            raise ValueError("T debe ser positivo y finito.")
        if initial not in self.index:
            raise ValueError("El estado inicial no pertenece al espacio de estados.")

        rng = np.random.default_rng(seed)
        t = 0.0
        state = initial

        times = [0.0]
        states = [state]
        events = ["inicio"]
        accepted = 0
        blocked = 0
        occupation = {s: 0.0 for s in self.states}

        while t < T:
            trans = self.transitions(state, include_blocked_arrival=True)
            rates = np.array([r for _, _, r in trans], dtype=float)
            total_rate = rates.sum()

            if total_rate <= 0:
                occupation[state] += T - t
                break

            dt = rng.exponential(1.0 / total_rate)
            if t + dt > T:
                occupation[state] += T - t
                t = T
                break

            occupation[state] += dt
            t += dt
            k = rng.choice(len(trans), p=rates / total_rate)
            event, target, _ = trans[k]

            if event == "llegada":
                accepted += 1
            elif event == "llegada_bloqueada":
                blocked += 1

            state = target
            times.append(t)
            states.append(state)
            events.append(event)

        history = pd.DataFrame(
            {
                "t": times,
                "estado": states,
                "N": [self.total_customers(s) for s in states],
                "en_espera": [self.waiting_customers(s) for s in states],
                "servidor_1_ocupado": [self.server1_busy(s) for s in states],
                "servidor_2_ocupado": [self.server2_busy(s) for s in states],
                "evento": events,
            }
        )

        empirical = pd.Series(
            [occupation[s] / T for s in self.states],
            index=self.states,
            name="pi_empirica",
        )

        stats = {
            "T": T,
            "llegadas_aceptadas": accepted,
            "llegadas_bloqueadas": blocked,
            "fraccion_bloqueada_eventos": blocked / max(accepted + blocked, 1),
        }
        return history, empirical, stats

    # ------------------------------------------------------------------
    # Barridos paramétricos
    # ------------------------------------------------------------------
    def lambda_sweep(self, lambdas: Iterable[float]) -> pd.DataFrame:
        rows = []
        for lam in lambdas:
            m = FiniteSingleQueueTwoServers(self.p.M, float(lam), self.p.mu1, self.p.mu2)
            rows.append({"lambda": float(lam), **m.metrics()})
        return pd.DataFrame(rows)

    def capacity_sweep(self, capacities: Iterable[int]) -> pd.DataFrame:
        rows = []
        for M in capacities:
            m = FiniteSingleQueueTwoServers(int(M), self.p.lam, self.p.mu1, self.p.mu2)
            rows.append({"M": int(M), **m.metrics()})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Gráficas
    # ------------------------------------------------------------------
    def plot_stationary_states(self, pi: pd.Series | None = None, ax=None):
        if pi is None:
            pi = self.stationary_distribution()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4.8))

        x = np.arange(len(self.states))
        ax.bar(x, pi.reindex(self.states).to_numpy(dtype=float))
        ax.set_xticks(x)
        ax.set_xticklabels(self.states, rotation=45 if len(self.states) > 12 else 0)
        ax.set_xlabel("Estado")
        ax.set_ylabel("Probabilidad estacionaria")
        ax.set_title("Distribución estacionaria por estado")
        return ax

    def plot_total_distribution(self, pi: pd.Series | None = None, ax=None):
        pN = self.total_distribution(pi)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(pN.index, pN.to_numpy(dtype=float))
        ax.set_xlabel("Número total de clientes N")
        ax.set_ylabel("P(N=n)")
        ax.set_title("Distribución estacionaria del número de clientes")
        ax.set_xticks(pN.index)
        return ax

    def plot_server_utilization(self, pi: pd.Series | None = None, ax=None):
        met = self.metrics(pi)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))
        vals = [met["utilizacion_servidor_1"], met["utilizacion_servidor_2"]]
        ax.bar(["Servidor 1", "Servidor 2"], vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fracción de tiempo ocupado")
        ax.set_title("Utilización estacionaria de los servidores")
        return ax

    def plot_sample_path(self, history: pd.DataFrame, ax=None, max_time: float | None = None):
        h = history
        if max_time is not None:
            h = h[h["t"] <= max_time]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4.8))
        ax.step(h["t"], h["N"], where="post", label="N(t)")
        ax.step(h["t"], h["en_espera"], where="post", label="Clientes esperando")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Clientes")
        ax.set_title("Trayectoria simulada")
        ax.legend()
        return ax

    def plot_lambda_sweep(self, lambdas: Iterable[float]):
        df = self.lambda_sweep(lambdas)

        fig1, ax1 = plt.subplots(figsize=(8, 4.8))
        ax1.plot(df["lambda"], df["E[N]"], marker="o", markersize=3)
        ax1.axvline(self.p.muT, linestyle="--", label=r"$\mu_1+\mu_2$ (referencia)")
        ax1.set_xlabel(r"$\lambda$")
        ax1.set_ylabel(r"$E[N]$")
        ax1.set_title("Número medio de clientes vs. tasa de llegada")
        ax1.legend()

        fig2, ax2 = plt.subplots(figsize=(8, 4.8))
        ax2.plot(df["lambda"], df["P_bloqueo"], marker="o", markersize=3)
        ax2.axvline(self.p.muT, linestyle="--", label=r"$\mu_1+\mu_2$ (referencia)")
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("Probabilidad de bloqueo")
        ax2.set_title("Probabilidad de bloqueo vs. tasa de llegada")
        ax2.legend()

        fig3, ax3 = plt.subplots(figsize=(8, 4.8))
        ax3.plot(df["lambda"], df["utilizacion_servidor_1"], label="Servidor 1")
        ax3.plot(df["lambda"], df["utilizacion_servidor_2"], label="Servidor 2")
        ax3.set_xlabel(r"$\lambda$")
        ax3.set_ylabel("Utilización")
        ax3.set_ylim(0, 1.02)
        ax3.set_title("Utilización de los servidores vs. tasa de llegada")
        ax3.legend()

        return df, (fig1, fig2, fig3)


def save_default_outputs(
    model: FiniteSingleQueueTwoServers,
    outdir: str | Path,
    T: float = 250.0,
    seed: int = 1234,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pi = model.stationary_distribution()
    files = []

    fig, ax = plt.subplots(figsize=(10, 4.8))
    model.plot_stationary_states(pi, ax=ax)
    fig.tight_layout()
    p = outdir / "01_distribucion_estacionaria_estados.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    model.plot_total_distribution(pi, ax=ax)
    fig.tight_layout()
    p = outdir / "02_distribucion_total_clientes.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    model.plot_server_utilization(pi, ax=ax)
    fig.tight_layout()
    p = outdir / "03_utilizacion_servidores.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    history, empirical, simstats = model.simulate(T=T, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    model.plot_sample_path(history, ax=ax, max_time=min(T, 60.0))
    fig.tight_layout()
    p = outdir / "04_trayectoria_simulada.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    # Barrido que cruza rho=1: útil para visualizar la diferencia entre caso finito e infinito.
    center = model.p.muT
    lambdas = np.linspace(0.08 * center, 2.0 * center, 45)
    sweep, figs = model.plot_lambda_sweep(lambdas)

    names = [
        "05_EN_vs_lambda.png",
        "06_bloqueo_vs_lambda.png",
        "07_utilizacion_vs_lambda.png",
    ]
    for fig, name in zip(figs, names):
        fig.tight_layout()
        p = outdir / name
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # Barrido en capacidad M.
    Ms = list(range(2, max(20, model.p.M + 1)))
    cap = model.capacity_sweep(Ms)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(cap["M"], cap["P_bloqueo"], marker="o", markersize=3)
    ax.set_xlabel("Capacidad total M")
    ax.set_ylabel("Probabilidad de bloqueo")
    ax.set_title("Efecto de la capacidad sobre el bloqueo")
    fig.tight_layout()
    p = outdir / "08_bloqueo_vs_M.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    model.probability_table(pi).to_csv(outdir / "probabilidades_estacionarias.csv", index=False)
    history.to_csv(outdir / "trayectoria_simulada.csv", index=False)
    sweep.to_csv(outdir / "barrido_lambda.csv", index=False)
    cap.to_csv(outdir / "barrido_M.csv", index=False)

    pd.DataFrame([model.metrics(pi)]).to_csv(outdir / "medidas_desempeno.csv", index=False)

    return files, history, empirical, simstats, sweep, cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo finito de una cola y dos servidores")
    parser.add_argument("--M", type=int, default=12, help="Capacidad total del sistema")
    parser.add_argument("--lam", type=float, default=4.0, help="Tasa de llegadas lambda")
    parser.add_argument("--mu1", type=float, default=3.0, help="Tasa del servidor 1")
    parser.add_argument("--mu2", type=float, default=2.0, help="Tasa del servidor 2")
    parser.add_argument("--T", type=float, default=250.0, help="Horizonte de simulación")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria")
    parser.add_argument("--outdir", default="salida_una_cola_dos_servidores", help="Directorio de salida")
    args = parser.parse_args()

    model = FiniteSingleQueueTwoServers(args.M, args.lam, args.mu1, args.mu2)
    pi = model.stationary_distribution()

    print("\nParámetros")
    print("----------")
    print(f"M      = {model.p.M}")
    print(f"lambda = {model.p.lam}")
    print(f"mu1    = {model.p.mu1}")
    print(f"mu2    = {model.p.mu2}")
    print(f"rho    = {model.p.rho:.10g}")

    print("\nDiagnóstico numérico")
    print("--------------------")
    for k, v in model.diagnostics(pi).items():
        print(f"{k:30s}: {v}")

    print("\nMedidas de desempeño")
    print("---------------------")
    for k, v in model.metrics(pi).items():
        print(f"{k:30s}: {v:.10g}")

    files, _, empirical, simstats, _, _ = save_default_outputs(
        model, args.outdir, T=args.T, seed=args.seed
    )

    exact = pi.reindex(model.states).to_numpy(dtype=float)
    emp = empirical.reindex(model.states).to_numpy(dtype=float)

    print("\nSimulación")
    print("----------")
    for k, v in simstats.items():
        print(f"{k:30s}: {v}")
    print(f"L1 empírica vs. exacta       : {np.sum(np.abs(emp - exact)):.6g}")

    print("\nArchivos generados")
    print("------------------")
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
