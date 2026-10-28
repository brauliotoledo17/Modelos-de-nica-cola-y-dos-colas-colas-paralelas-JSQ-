"""Modelo finito de dos colas paralelas con política JSQ y cambios entre colas.

Estado: X(t) = (Q1(t), Q2(t)), con capacidad M en cada cola.
Reglas:
  * Llegadas Poisson de tasa lam.
  * Servicio exponencial con tasas mu y mu2.
  * JSQ: una llegada entra a la cola más corta; en empate entra a Q1.
  * Si tras una salida las longitudes difieren en 2, el último cliente de la
    cola larga cambia instantáneamente a la corta.
  * En (M,M) las llegadas se rechazan.

El cálculo principal de la distribución estacionaria usa las fórmulas cerradas
normalizadas en escala logarítmica (log-sum-exp), evitando overflow/underflow
intermedio cuando rho^(2M) es muy grande o muy pequeño. También se construye el
generador Q de la CTMC para verificar numéricamente pi Q = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

State = Tuple[int, int]


@dataclass(frozen=True)
class Parameters:
    M: int
    lam: float
    mu: float
    mu2: float

    def __post_init__(self) -> None:
        if not isinstance(self.M, (int, np.integer)) or self.M < 1:
            raise ValueError("M debe ser un entero >= 1.")
        for name, value in (("lambda", self.lam), ("mu", self.mu), ("mu'", self.mu2)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} debe ser positivo y finito.")

    @property
    def rho(self) -> float:
        return self.lam / (self.mu + self.mu2)

    @property
    def A(self) -> float:
        return self.lam + self.mu + self.mu2


class FiniteJSQModel:
    """CTMC finita del modelo JSQ con intercambio instantáneo."""

    def __init__(self, M: int, lam: float, mu: float, mu2: float):
        self.p = Parameters(M=M, lam=lam, mu=mu, mu2=mu2)
        self.states: List[State] = self._build_states()
        self.index: Dict[State, int] = {s: k for k, s in enumerate(self.states)}

    # ------------------------------------------------------------------
    # Espacio de estados y dinámica
    # ------------------------------------------------------------------
    def _build_states(self) -> List[State]:
        M = self.p.M
        states = [(x, x) for x in range(M + 1)]
        states += [(x + 1, x) for x in range(M)]
        states += [(x, x + 1) for x in range(M)]
        # Orden útil para tablas/gráficas: total de clientes, luego Q1 y Q2.
        return sorted(set(states), key=lambda s: (s[0] + s[1], s[0], s[1]))

    @staticmethod
    def _rebalance(i: int, j: int) -> State:
        """Aplica como máximo un cambio instantáneo para restaurar |i-j| <= 1."""
        if i - j >= 2:
            i -= 1
            j += 1
        elif j - i >= 2:
            i += 1
            j -= 1
        return i, j

    def arrival_target(self, state: State) -> State:
        i, j = state
        M = self.p.M
        if state == (M, M):
            return state  # llegada bloqueada
        if i <= j:  # Q1 es más corta o hay empate; el empate va a Q1
            i += 1
        else:
            j += 1
        return self._rebalance(i, j)

    def service1_target(self, state: State) -> State | None:
        i, j = state
        if i == 0:
            return None
        return self._rebalance(i - 1, j)

    def service2_target(self, state: State) -> State | None:
        i, j = state
        if j == 0:
            return None
        return self._rebalance(i, j - 1)

    def transitions(self, state: State, include_blocked_arrival: bool = False):
        """Lista (evento, destino, tasa) desde un estado."""
        out = []
        a = self.arrival_target(state)
        if a != state or include_blocked_arrival:
            out.append(("llegada" if a != state else "llegada_bloqueada", a, self.p.lam))

        s1 = self.service1_target(state)
        if s1 is not None:
            out.append(("servicio_Q1", s1, self.p.mu))

        s2 = self.service2_target(state)
        if s2 is not None:
            out.append(("servicio_Q2", s2, self.p.mu2))
        return out

    # ------------------------------------------------------------------
    # Generador Q
    # ------------------------------------------------------------------
    def generator(self) -> csr_matrix:
        """Construye el generador Q de la CTMC (sin auto-eventos bloqueados)."""
        n = len(self.states)
        rows, cols, data = [], [], []

        for state in self.states:
            r = self.index[state]
            rates: Dict[State, float] = {}
            for _, target, rate in self.transitions(state, include_blocked_arrival=False):
                if target == state:
                    continue
                rates[target] = rates.get(target, 0.0) + rate

            total = 0.0
            for target, rate in rates.items():
                rows.append(r)
                cols.append(self.index[target])
                data.append(rate)
                total += rate

            rows.append(r)
            cols.append(r)
            data.append(-total)

        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)

    # ------------------------------------------------------------------
    # Distribución estacionaria estable
    # ------------------------------------------------------------------
    def _log_unnormalized_weights(self) -> np.ndarray:
        """Logaritmos de los pesos estacionarios no normalizados.

        Se calculan sin formar rho**(2*x), lo que evita overflow intermedio.
        """
        lam, mu, mu2 = self.p.lam, self.p.mu, self.p.mu2
        A = self.p.A
        log_rho = np.log(lam) - np.log(mu + mu2)
        logA = np.log(A)

        logw = np.full(len(self.states), -np.inf, dtype=float)

        def put(state: State, value: float) -> None:
            logw[self.index[state]] = value

        # K0, K1, K2 del teorema, escritos directamente en logaritmos.
        logK0 = (
            2.0 * log_rho
            + np.log(mu)
            + np.log(mu2)
            - 2.0 * np.log(lam)
            + np.log(A + lam)
            - np.log(mu2 + lam)
        )
        logK1 = np.log(mu) - np.log(mu2 + lam) + 2.0 * log_rho
        logK2 = (
            np.log(mu2)
            + logA
            - np.log(lam)
            - np.log(mu2 + lam)
            + 2.0 * log_rho
        )

        put((0, 0), logK0)
        put((0, 1), logK1)
        put((1, 0), logK2)

        # Diagonal: rho^(2x)
        for x in range(1, self.p.M + 1):
            put((x, x), 2.0 * x * log_rho)

        # Estados adyacentes a la diagonal, x = 1,...,M-1.
        # (x, x+1): (mu/A) rho^(2(x+1))
        # (x+1, x): rho^(2x) [lambda + mu' rho^2]/A
        log_mix = np.logaddexp(np.log(lam), np.log(mu2) + 2.0 * log_rho)
        for x in range(1, self.p.M):
            put((x, x + 1), np.log(mu) - logA + 2.0 * (x + 1) * log_rho)
            put((x + 1, x), 2.0 * x * log_rho + log_mix - logA)

        if not np.all(np.isfinite(logw)):
            raise FloatingPointError("Se generaron pesos logarítmicos no finitos.")
        return logw

    def log_stationary_distribution(self) -> pd.Series:
        """Devuelve log(pi) sin perder probabilidades extremadamente pequeñas.

        Esta representación sigue siendo útil incluso cuando una probabilidad es tan
        pequeña que no puede representarse como float64 y exp(log(pi)) se redondea a 0.
        """
        logw = self._log_unnormalized_weights()
        logpi = logw - logsumexp(logw)
        return pd.Series(logpi, index=pd.Index(self.states, dtype="object"), name="log_pi")

    def stationary_distribution(self) -> pd.Series:
        """Distribución estacionaria usando normalización log-sum-exp."""
        logpi = self.log_stationary_distribution().to_numpy(dtype=float)
        pi = np.exp(logpi)
        # Renormalización final para absorber redondeo de máquina.
        pi /= pi.sum(dtype=np.float64)
        return pd.Series(pi, index=pd.Index(self.states, dtype="object"), name="pi")

    def stationary_via_generator(self) -> pd.Series:
        """Resuelve pi Q = 0 + sum(pi)=1 como verificación independiente.

        Q se escala para evitar que tasas muy grandes o pequeñas empeoren el
        condicionamiento. Esta ruta es útil como control, no como método principal
        para M extremadamente grande.
        """
        Q = self.generator().astype(float)
        n = Q.shape[0]
        scale = max(self.p.A, 1.0)
        Aeq = (Q.T / scale).tolil()
        Aeq[-1, :] = np.ones(n)
        Aeq = csc_matrix(Aeq)
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.asarray(spsolve(Aeq, b), dtype=float)

        # Sólo toleramos negativos de redondeo minúsculos.
        tol = 1e-11
        if np.min(pi) < -tol:
            raise FloatingPointError(
                f"La solución por generador produjo una probabilidad negativa: {pi.min():.3e}"
            )
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum()
        return pd.Series(pi, index=pd.Index(self.states, dtype="object"), name="pi_generator")

    def diagnostics(self, pi: pd.Series | None = None) -> dict:
        if pi is None:
            pi = self.stationary_distribution()
        vec = pi.reindex(self.states).to_numpy(dtype=float)
        Q = self.generator()
        residual = np.asarray(vec @ Q).ravel()

        result = {
            "num_states": len(self.states),
            "sum_pi": float(vec.sum()),
            "min_pi": float(vec.min()),
            "max_balance_residual": float(np.max(np.abs(residual))),
        }

        # Comparación independiente cuando el tamaño es razonable.
        if len(self.states) <= 5000:
            pi_q = self.stationary_via_generator()
            vq = pi_q.reindex(self.states).to_numpy(dtype=float)
            result["L1_formula_vs_generator"] = float(np.sum(np.abs(vec - vq)))
            result["Linf_formula_vs_generator"] = float(np.max(np.abs(vec - vq)))
        return result

    # ------------------------------------------------------------------
    # Medidas de desempeño
    # ------------------------------------------------------------------
    def metrics(self, pi: pd.Series | None = None) -> dict:
        if pi is None:
            pi = self.stationary_distribution()
        probs = pi.reindex(self.states).to_numpy(dtype=float)
        q1 = np.array([s[0] for s in self.states], dtype=float)
        q2 = np.array([s[1] for s in self.states], dtype=float)
        p_block = float(pi.loc[(self.p.M, self.p.M)])

        return {
            "rho": self.p.rho,
            "E[Q1]": float(probs @ q1),
            "E[Q2]": float(probs @ q2),
            "E[N]": float(probs @ (q1 + q2)),
            "P_bloqueo": p_block,
            "tasa_aceptada": self.p.lam * (1.0 - p_block),
            "utilizacion_Q1": float(probs @ (q1 > 0)),
            "utilizacion_Q2": float(probs @ (q2 > 0)),
        }

    def probability_table(self, pi: pd.Series | None = None) -> pd.DataFrame:
        if pi is None:
            pi = self.stationary_distribution()
        rows = []
        for state in self.states:
            rows.append({
                "Q1": state[0],
                "Q2": state[1],
                "N": state[0] + state[1],
                "pi": float(pi.loc[state]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Simulación Gillespie / eventos
    # ------------------------------------------------------------------
    def simulate(self, T: float, seed: int | None = None, initial: State = (0, 0)):
        if not np.isfinite(T) or T <= 0:
            raise ValueError("T debe ser positivo y finito.")
        if initial not in self.index:
            raise ValueError("El estado inicial no pertenece a S_M.")

        rng = np.random.default_rng(seed)
        t = 0.0
        state = initial
        times = [0.0]
        states = [state]
        events = ["inicio"]
        accepted = 0
        blocked = 0

        # Tiempo acumulado en cada estado para una distribución empírica temporal.
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

            if event == "llegada_bloqueada":
                blocked += 1
            elif event == "llegada":
                accepted += 1

            state = target
            times.append(t)
            states.append(state)
            events.append(event)

        history = pd.DataFrame({
            "t": times,
            "Q1": [s[0] for s in states],
            "Q2": [s[1] for s in states],
            "N": [s[0] + s[1] for s in states],
            "evento": events,
        })
        empirical = pd.Series(
            [occupation[s] / T for s in self.states],
            index=pd.Index(self.states, dtype="object"),
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
    # Gráficas
    # ------------------------------------------------------------------
    def plot_stationary_heatmap(self, pi: pd.Series | None = None, ax=None):
        if pi is None:
            pi = self.stationary_distribution()
        M = self.p.M
        mat = np.full((M + 1, M + 1), np.nan)
        for (i, j), prob in pi.items():
            mat[j, i] = prob  # eje x = Q1, eje y = Q2

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, origin="lower", aspect="equal")
        ax.set_xlabel("Q1")
        ax.set_ylabel("Q2")
        ax.set_title("Distribución estacionaria sobre $S_M$")
        plt.colorbar(im, ax=ax, label="Probabilidad estacionaria")
        return ax

    def plot_total_customers(self, pi: pd.Series | None = None, ax=None):
        if pi is None:
            pi = self.stationary_distribution()
        totals = {}
        for state, prob in pi.items():
            n = state[0] + state[1]
            totals[n] = totals.get(n, 0.0) + float(prob)
        x = np.array(sorted(totals))
        y = np.array([totals[k] for k in x])

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, y)
        ax.set_xlabel("Número total de clientes N = Q1 + Q2")
        ax.set_ylabel("Probabilidad")
        ax.set_title("Distribución estacionaria del número total de clientes")
        ax.set_xticks(x)
        return ax

    def plot_queue_marginals(self, pi: pd.Series | None = None, ax=None):
        if pi is None:
            pi = self.stationary_distribution()
        M = self.p.M
        q1 = np.zeros(M + 1)
        q2 = np.zeros(M + 1)
        for (i, j), prob in pi.items():
            q1[i] += prob
            q2[j] += prob

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(M + 1)
        ax.plot(x, q1, marker="o", label="Q1")
        ax.plot(x, q2, marker="s", label="Q2")
        ax.set_xlabel("Número de clientes")
        ax.set_ylabel("Probabilidad marginal")
        ax.set_title("Distribuciones marginales de las colas")
        ax.legend()
        return ax

    def plot_sample_path(self, history: pd.DataFrame, ax=None, max_time: float | None = None):
        h = history
        if max_time is not None:
            h = h[h["t"] <= max_time]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4.5))
        ax.step(h["t"], h["Q1"], where="post", label="Q1")
        ax.step(h["t"], h["Q2"], where="post", label="Q2")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Clientes")
        ax.set_title("Trayectoria simulada del sistema")
        ax.legend()
        return ax

    def lambda_sweep(self, lambdas: Iterable[float]) -> pd.DataFrame:
        rows = []
        for lam in lambdas:
            m = FiniteJSQModel(self.p.M, float(lam), self.p.mu, self.p.mu2)
            met = m.metrics()
            rows.append({"lambda": lam, **met})
        return pd.DataFrame(rows)

    def plot_lambda_sweep(self, lambdas: Iterable[float]):
        df = self.lambda_sweep(lambdas)

        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(df["lambda"], df["E[N]"], marker="o")
        ax1.set_xlabel(r"$\lambda$")
        ax1.set_ylabel(r"$E[N]$")
        ax1.set_title("Clientes promedio vs. tasa de llegada")
        ax1.axvline(self.p.mu + self.p.mu2, linestyle="--", label=r"$\mu+\mu'$ (referencia)")
        ax1.legend()

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(df["lambda"], df["P_bloqueo"], marker="o")
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("Probabilidad de bloqueo")
        ax2.set_title("Bloqueo vs. tasa de llegada")

        return df, (fig1, fig2)


def save_default_plots(model: FiniteJSQModel, outdir: str | Path, T: float = 100.0, seed: int = 1234):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pi = model.stationary_distribution()

    figures = []

    fig, ax = plt.subplots(figsize=(7, 6))
    model.plot_stationary_heatmap(pi, ax=ax)
    fig.tight_layout()
    path = outdir / "01_distribucion_estacionaria.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    model.plot_total_customers(pi, ax=ax)
    fig.tight_layout()
    path = outdir / "02_total_clientes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    model.plot_queue_marginals(pi, ax=ax)
    fig.tight_layout()
    path = outdir / "03_marginales_colas.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    history, empirical, simstats = model.simulate(T=T, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    model.plot_sample_path(history, ax=ax)
    fig.tight_layout()
    path = outdir / "04_trayectoria_simulada.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    lam_center = model.p.mu + model.p.mu2
    lambdas = np.linspace(0.1 * lam_center, 2.0 * lam_center, 35)
    sweep, (fig1, fig2) = model.plot_lambda_sweep(lambdas)
    path1 = outdir / "05_EN_vs_lambda.png"
    fig1.tight_layout()
    fig1.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    figures.append(path1)

    path2 = outdir / "06_bloqueo_vs_lambda.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    figures.append(path2)

    model.probability_table(pi).to_csv(outdir / "probabilidades_estacionarias.csv", index=False)
    history.to_csv(outdir / "trayectoria_simulada.csv", index=False)
    sweep.to_csv(outdir / "barrido_lambda.csv", index=False)

    return figures, history, empirical, simstats


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo finito JSQ con cambios entre colas")
    parser.add_argument("--M", type=int, default=10, help="Capacidad máxima por cola")
    parser.add_argument("--lam", type=float, default=4.0, help="Tasa de llegadas lambda")
    parser.add_argument("--mu", type=float, default=3.0, help="Tasa de servicio de Q1")
    parser.add_argument("--mu2", type=float, default=2.0, help="Tasa de servicio de Q2 (mu')")
    parser.add_argument("--T", type=float, default=100.0, help="Horizonte de simulación")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria")
    parser.add_argument("--outdir", default="salida_jsq", help="Directorio de salida")
    args = parser.parse_args()

    model = FiniteJSQModel(args.M, args.lam, args.mu, args.mu2)
    pi = model.stationary_distribution()

    print("\nParámetros")
    print("----------")
    print(f"M      = {model.p.M}")
    print(f"lambda = {model.p.lam}")
    print(f"mu     = {model.p.mu}")
    print(f"mu'    = {model.p.mu2}")
    print(f"rho    = {model.p.rho:.8g}")

    print("\nDiagnóstico numérico")
    print("--------------------")
    for k, v in model.diagnostics(pi).items():
        print(f"{k:28s}: {v}")

    print("\nMedidas de desempeño")
    print("---------------------")
    for k, v in model.metrics(pi).items():
        print(f"{k:28s}: {v:.10g}")

    figures, _, empirical, simstats = save_default_plots(
        model, args.outdir, T=args.T, seed=args.seed
    )

    exact = pi.reindex(model.states).to_numpy()
    emp = empirical.reindex(model.states).to_numpy()
    print("\nSimulación")
    print("----------")
    for k, v in simstats.items():
        print(f"{k:28s}: {v}")
    print(f"L1 empírica vs. exacta       : {np.sum(np.abs(emp - exact)):.6g}")

    print("\nArchivos generados")
    print("------------------")
    for f in figures:
        print(f)
    print(Path(args.outdir) / "probabilidades_estacionarias.csv")
    print(Path(args.outdir) / "trayectoria_simulada.csv")
    print(Path(args.outdir) / "barrido_lambda.csv")


if __name__ == "__main__":
    main()
