"""
Parte B - Simulação e validação do Filtro de Kalman aplicado à fusão
giroscópio + acelerômetro para estimação de ângulo de inclinação (pitch).

Modelo de estado usado (clássico para atitude com IMU):

    x = [theta, bias]^T   -> ângulo (deg) e viés (bias) do giroscópio (deg/s)

    Predição (a priori, modelo dinâmico do giroscópio):
        theta_k = theta_{k-1} + dt * (gyro_k - bias_{k-1})
        bias_k  = bias_{k-1}
        F = [[1, -dt], [0, 1]] ,  B = [[dt], [0]] ,  u_k = gyro_k

    Atualização (verossimilhança da medição do acelerômetro):
        z_k = theta_k + ruído
        H = [1, 0]

Gera sinal sintético de inclinação real, simula leituras ruidosas de
giroscópio (integradas -> deriva) e acelerômetro (ruidosas mas sem deriva),
roda o filtro, plota os resultados e calcula o EQM (MSE) para validar o
ganho estatístico do filtro frente às medições brutas.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from kalman import KalmanFilter

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def gerar_sinal_real(t, amplitude_deg=30.0, freq_hz=0.15):
    """Sinal senoidal representando a inclinação real do corpo rígido."""
    theta = amplitude_deg * np.sin(2 * np.pi * freq_hz * t)
    theta_dot = amplitude_deg * 2 * np.pi * freq_hz * np.cos(2 * np.pi * freq_hz * t)
    return theta, theta_dot


def simular_giroscopio(theta_dot, bias_real, sigma_gyro, rng):
    """Giroscópio: mede a taxa angular real, com viés constante + ruído branco de baixa variância."""
    ruido = rng.normal(0.0, sigma_gyro, size=theta_dot.shape)
    return theta_dot + bias_real + ruido


def simular_acelerometro(theta, sigma_accel, rng):
    """Acelerômetro: mede o ângulo diretamente, com ruído gaussiano de alta variância."""
    ruido = rng.normal(0.0, sigma_accel, size=theta.shape)
    return theta + ruido


def calcular_eqm(estimado, real):
    return float(np.mean((np.asarray(estimado) - np.asarray(real)) ** 2))


def main():
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------
    # 1) Sinal real (verdade de referência)
    # ---------------------------------------------------------------
    dt = 0.01
    T = 20.0
    t = np.arange(0, T, dt)
    theta_real, theta_dot_real = gerar_sinal_real(t, amplitude_deg=45.0, freq_hz=0.25)

    # ---------------------------------------------------------------
    # 2) Sensores simulados
    # ---------------------------------------------------------------
    bias_real = 3.5           # deg/s, viés constante do giroscópio
    sigma_gyro = 0.3          # deg/s, ruído de baixa variância (giroscópio)
    sigma_accel = 6.0         # deg,   ruído de alta variância (acelerômetro)

    gyro_medido = simular_giroscopio(theta_dot_real, bias_real, sigma_gyro, rng)
    accel_medido = simular_acelerometro(theta_real, sigma_accel, rng)

    # ---------------------------------------------------------------
    # 3) Configuração do Filtro de Kalman
    # ---------------------------------------------------------------
    F = [[1.0, -dt],
         [0.0, 1.0]]
    B = [[dt],
         [0.0]]
    H = [[1.0, 0.0]]

    # Q: incerteza injetada no modelo de processo (giroscópio integrado).
    q_theta = (sigma_gyro * dt) ** 2
    q_bias = 1e-4
    Q = [[q_theta, 0.0],
         [0.0, q_bias]]

    # R: incerteza da medição do acelerômetro.
    R = [[sigma_accel ** 2]]

    x0 = [0.0, 0.0]
    P0 = [[5.0, 0.0],
          [0.0, 5.0]]

    kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, P0=P0, x0=x0)

    # ---------------------------------------------------------------
    # 4) Execução do filtro amostra a amostra
    # ---------------------------------------------------------------
    theta_filtrado = np.zeros_like(t)
    bias_estimado = np.zeros_like(t)

    for k in range(len(t)):
        kf.predict(u=[gyro_medido[k]])
        kf.update(z=[accel_medido[k]])
        theta_filtrado[k] = kf.state[0]
        bias_estimado[k] = kf.state[1]

    # ---------------------------------------------------------------
    # 5) Validação estatística (EQM)
    # ---------------------------------------------------------------
    eqm_accel = calcular_eqm(accel_medido, theta_real)
    eqm_filtrado = calcular_eqm(theta_filtrado, theta_real)
    ganho_percentual = 100 * (1 - eqm_filtrado / eqm_accel)

    print("=== Validação estatística (EQM) ===")
    print(f"EQM acelerômetro (medição bruta) : {eqm_accel:.4f} deg^2")
    print(f"EQM Filtro de Kalman             : {eqm_filtrado:.4f} deg^2")
    print(f"Ganho estatístico do filtro       : {ganho_percentual:.2f}%")
    print(f"Viés real do giroscópio: {bias_real:.2f} deg/s | "
          f"Viés estimado (final): {bias_estimado[-1]:.2f} deg/s")

    # ---------------------------------------------------------------
    # 6) Gráficos
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, theta_real, label="Real", color="black", linewidth=2)
    axes[0].plot(t, accel_medido, label="Medido (acelerômetro)", color="tab:red",
                 alpha=0.4, linewidth=0.8)
    axes[0].plot(t, theta_filtrado, label="Filtrado (Kalman)", color="tab:blue",
                 linewidth=1.5)
    axes[0].set_ylabel("Ângulo (graus)")
    axes[0].set_title("Real vs. Medido vs. Filtrado")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, theta_dot_real, label="Taxa angular real", color="black", linewidth=2)
    axes[1].plot(t, gyro_medido, label="Giroscópio (medido)", color="tab:orange",
                 alpha=0.6, linewidth=0.8)
    axes[1].set_ylabel("Taxa angular (graus/s)")
    axes[1].set_title("Giroscópio: real vs. medido")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, np.full_like(t, bias_real), "--", color="black", label="Viés real")
    axes[2].plot(t, bias_estimado, color="tab:green", label="Viés estimado (Kalman)")
    axes[2].set_ylabel("Viés (graus/s)")
    axes[2].set_xlabel("Tempo (s)")
    axes[2].set_title("Estimação do viés do giroscópio")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kalman_imu_simulation.png", dpi=150)

    # Gráfico auxiliar do erro instantâneo (medição bruta vs. filtrado)
    fig_err, ax_err = plt.subplots(figsize=(10, 4))
    ax_err.plot(t, np.abs(accel_medido - theta_real), color="tab:red", alpha=0.5,
                label="Erro absoluto - acelerômetro")
    ax_err.plot(t, np.abs(theta_filtrado - theta_real), color="tab:blue",
                label="Erro absoluto - Kalman")
    ax_err.set_xlabel("Tempo (s)")
    ax_err.set_ylabel("Erro absoluto (graus)")
    ax_err.set_title("Comparação do erro: medição bruta vs. filtro de Kalman")
    ax_err.legend(loc="upper right")
    ax_err.grid(True, alpha=0.3)
    fig_err.tight_layout()
    fig_err.savefig(OUTPUT_DIR / "kalman_imu_error.png", dpi=150)

    print(f"\nGráficos salvos em: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
