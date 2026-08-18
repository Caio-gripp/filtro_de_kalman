"""
Parte A - Filtro de Kalman Linear genérico.

Implementação usando exclusivamente numpy para álgebra matricial.
A classe não conhece nada sobre o problema de fusão de IMU: ela apenas
implementa as equações de predição (a priori) e atualização (a posteriori)
do Filtro de Kalman para um sistema linear-gaussiano genérico:

    x_k = F x_{k-1} + B u_k + w_k ,   w_k ~ N(0, Q)   (modelo de processo)
    z_k = H x_k + v_k ,               v_k ~ N(0, R)   (modelo de medição)
"""

import numpy as np


class KalmanFilter:
    def __init__(self, F, B, H, Q, R, P0, x0):
        """
        Parâmetros
        ----------
        F  : (n, n) matriz de transição de estados
        B  : (n, m) matriz de controle (pode ser None se não houver entrada u)
        H  : (p, n) matriz de observação
        Q  : (n, n) covariância do ruído de processo
        R  : (p, p) covariância do ruído de medição
        P0 : (n, n) covariância inicial do estado
        x0 : (n,) ou (n, 1) estado inicial
        """
        self.F = np.atleast_2d(F).astype(float)
        self.B = np.atleast_2d(B).astype(float) if B is not None else None
        self.H = np.atleast_2d(H).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)

        self.P = np.atleast_2d(P0).astype(float)
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)

        n = self.x.shape[0]
        self._I = np.eye(n)

    def predict(self, u=None):
        """Etapa de predição (a priori): propaga estado e covariância no tempo."""
        if self.B is not None and u is not None:
            u = np.asarray(u, dtype=float).reshape(-1, 1)
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        """Etapa de atualização (a posteriori): corrige o estado com a medição z."""
        z = np.asarray(z, dtype=float).reshape(-1, 1)

        y = z - self.H @ self.x                      # inovação (resíduo)
        S = self.H @ self.P @ self.H.T + self.R       # covariância da inovação
        K = self.P @ self.H.T @ np.linalg.inv(S)      # ganho de Kalman

        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        return self.x, self.P

    @property
    def state(self):
        """Retorna o estado atual como vetor 1D (n,)."""
        return self.x.flatten()


if __name__ == "__main__":
    # Demonstração mínima de genericidade: estimar uma posição escalar
    # constante e desconhecida a partir de medições ruidosas, SEM entrada
    # de controle (B=None, u=None). Mostra que a classe não é presa ao
    # problema da IMU.
    rng = np.random.default_rng(0)
    valor_real = 10.0
    medicoes = valor_real + rng.normal(0, 1.0, size=50)

    kf = KalmanFilter(
        F=[[1.0]], B=None, H=[[1.0]],
        Q=[[1e-5]], R=[[1.0]],
        P0=[[1.0]], x0=[0.0],
    )

    estimativas = []
    for z in medicoes:
        kf.predict()
        kf.update(z)
        estimativas.append(kf.state[0])

    print(f"Valor real: {valor_real}")
    print(f"Estimativa final: {estimativas[-1]:.4f}")
