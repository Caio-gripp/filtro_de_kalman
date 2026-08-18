Trabalho desenvolvido por Caio Gripp (202601203) e Luiza Kerscher (202501128).

Tentamos bastante fazer a parte C mas não conseguimos :(


# Filtro de Kalman

Implementação de um Filtro de Kalman linear genérico e sua aplicação à fusão de sensores de uma IMU (giroscópio + acelerômetro) para estimação do ângulo de inclinação (*pitch*) de um corpo rígido.

## Estrutura do projeto

```
.
├── src/
│   ├── kalman.py             # Parte A - Filtro de Kalman linear genérico
│   └── simulacao.py          # Parte B - Simulação de IMU e validação estatística
├── outputs/
│   ├── kalman_imu_simulation.png   # Real vs. medido vs. filtrado + estimação do viés
│   └── kalman_imu_error.png        # Erro absoluto: medição bruta vs. Kalman
├── main.pdf                 # Relatório
└── requirements.txt
```

## Parte A — Filtro de Kalman genérico

`src/kalman.py` implementa a classe `KalmanFilter`, usando apenas `numpy` para álgebra matricial. A classe é agnóstica ao problema de aplicação: implementa somente as equações de predição (a priori) e atualização (a posteriori) de um sistema linear-gaussiano genérico:

```
x_k = F x_{k-1} + B u_k + w_k ,   w_k ~ N(0, Q)   (modelo de processo)
z_k = H x_k + v_k ,               v_k ~ N(0, R)   (modelo de medição)
```

### Uso básico

```python
from kalman import KalmanFilter

kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, P0=P0, x0=x0)

kf.predict(u=u_k)   # etapa de predição (a priori)
kf.update(z=z_k)    # etapa de atualização (a posteriori)

estado_atual = kf.state
```

Executar o módulo diretamente (`python kalman.py`) roda uma demonstração mínima: estimação de uma posição escalar constante a partir de medições ruidosas, sem entrada de controle, mostrando que a classe não é presa ao problema da IMU.

## Parte B — Fusão giroscópio + acelerômetro

`src/simulacao.py` aplica o filtro ao problema clássico de fusão de atitude com IMU, com vetor de estado:

```
x = [theta, bias]^T
```

onde `theta` é o ângulo de inclinação (graus) e `bias` é o viés do giroscópio (graus/s).

- **Predição**: integra a taxa angular medida pelo giroscópio (`theta_k = theta_{k-1} + dt * (gyro_k - bias_{k-1})`), que sofre deriva ao longo do tempo.
- **Atualização**: corrige o ângulo com a leitura do acelerômetro, que é ruidosa mas não deriva.

O script:

1. Gera um sinal senoidal de referência representando a inclinação real do corpo rígido.
2. Simula leituras ruidosas de giroscópio (com viés constante) e acelerômetro (ruído de alta variância).
3. Executa o filtro amostra a amostra.
4. Calcula o Erro Quadrático Médio (EQM) da medição bruta do acelerômetro vs. da estimativa filtrada, para validar o ganho estatístico do filtro.
5. Gera e salva os gráficos em `outputs/`.

## Instalação

```bash
pip install -r requirements.txt
```

Dependências: `numpy`, `matplotlib`.

## Execução

```bash
cd src
python simulacao.py
```

O script imprime no console o EQM da medição bruta, o EQM do filtro de Kalman, o ganho estatístico percentual e o viés estimado do giroscópio, além de salvar os gráficos em `outputs/`:

- **`kalman_imu_simulation.png`**: comparação entre ângulo real, medido e filtrado; taxa angular real vs. medida; e estimação do viés do giroscópio ao longo do tempo.
- **`kalman_imu_error.png`**: comparação do erro absoluto entre a medição bruta do acelerômetro e a estimativa do filtro de Kalman.
