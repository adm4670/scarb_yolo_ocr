
# Calibração de Medição de Corrente com ESP32 usando Machine Learning

## 1. Contexto e Problema

Sistemas baseados em **ESP32 + sensor de corrente não invasivo (CT)** frequentemente apresentam divergências em relação a medições feitas por **alicates amperímetros True RMS**. Essas diferenças não são apenas erros lineares simples, mas resultam de uma combinação de fatores físicos, elétricos e computacionais.

Principais causas do erro:

* Não linearidade do sensor CT
* Saturação do núcleo magnético
* Offset e ruído do ADC do ESP32
* Amostragem insuficiente
* Cargas não lineares (fontes chaveadas, motores, inversores)
* Harmônicos e distorções da forma de onda
* Variações térmicas

Diante disso, uma calibração clássica (fator fixo ou curva simples) é insuficiente. A proposta é utilizar **Machine Learning para calibração não linear multivariada**, aprendendo a mapear a medição do ESP32 para o valor real observado no alicate amperímetro.

---

## 2. Formulação do Problema

O objetivo não é estimar corrente do zero, mas **corrigir** a leitura:

```
I_alicate = f(features_extraídas_do_sinal)
```

* **Target (y):** Corrente medida pelo alicate amperímetro
* **Inputs (X):** Features extraídas da forma de onda medida pelo ESP32

Trata-se de um problema de **regressão supervisionada**, com forte não linearidade.

---

## 3. Estratégia Geral

1. Capturar a forma de onda da corrente no ESP32
2. Extrair features relevantes (tempo, forma e contexto)
3. Sincronizar cada amostra com a leitura do alicate amperímetro
4. Treinar um modelo de regressão offline
5. Embarcar o modelo (ou aproximação dele) no ESP32

---

## 4. Aquisição do Sinal

### Parâmetros recomendados

* **Frequência da rede:** 50/60 Hz
* **Taxa de amostragem (Fs):** 4 kHz a 8 kHz
* **Janela de análise:** 1 a 2 ciclos da rede

  * 1 ciclo @ 60 Hz ≈ 67 amostras (4 kHz)
  * Ideal: 128–256 amostras

A forma de onda deve ser centrada em zero (remoção de offset DC).

---

## 5. Features a serem Calculadas no ESP32

As features foram escolhidas considerando **alto poder informativo** e **baixo custo computacional**.

### 5.1 Grupo 1 — Features Básicas (Obrigatórias)

| Feature      | Descrição            | Fórmula / Observação |
| ------------ | -------------------- | -------------------- |
| `Irms`       | RMS real da corrente | `sqrt(mean(i²))`     |
| `Ipeak`      | Pico absoluto        | `max(abs(i))`        |
| `Ipp`        | Pico a pico          | `max(i) - min(i)`    |
| `I_mean_abs` | Média absoluta       | `mean(abs(i))`       |
| `std`        | Desvio padrão        | Dispersão do sinal   |

Essas capturam amplitude, espalhamento e parte da distorção.

---

### 5.2 Grupo 2 — Forma de Onda (Alto Impacto)

| Feature         | Descrição                  | Motivação                             |
| --------------- | -------------------------- | ------------------------------------- |
| `crest_factor`  | `Ipeak / Irms`             | Identifica cargas impulsivas          |
| `shape_factor`  | `Irms / I_mean_abs`        | Diferencia senoide de ondas achatadas |
| `asymmetry`     | Diferença entre semiciclos | Detecta saturação e offset            |
| `pos_neg_ratio` | Pico positivo / negativo   | Assimetria de carga                   |

Essas features permitem ao modelo inferir **tipo de carga** e **grau de distorção**.

---

### 5.3 Grupo 3 — Frequência e Harmônicos (Sem FFT pesada)

| Feature         | Descrição                  | Implementação                  |
| --------------- | -------------------------- | ------------------------------ |
| `frequency`     | Frequência estimada        | Zero-crossing                  |
| `period_jitter` | Desvio do período          | std do tempo entre cruzamentos |
| `hf_energy`     | Energia de alta frequência | `mean((i[n]-i[n-1])²)`         |

`hf_energy` funciona como um **proxy barato de THD**, extremamente eficaz.

---

### 5.4 Grupo 4 — Contexto Físico

| Feature          | Descrição                      |
| ---------------- | ------------------------------ |
| `temperature`    | Temperatura do ESP32 ou sensor |
| `Vcc` (opcional) | Tensão de alimentação          |

Essas variáveis ajudam a capturar **drift lento e variações do ADC**.

---

### 5.5 Grupo 5 — Histórico (Estabilidade)

| Feature        | Descrição             |
| -------------- | --------------------- |
| `Irms_mean_5s` | Média móvel           |
| `Irms_std_5s`  | Variabilidade recente |

Essas reduzem ruído e evitam correções excessivas em transientes.

---

## 6. Feature Set Recomendado

### MVP (12 features)

```
Irms
Ipeak
Ipp
I_mean_abs
std
crest_factor
shape_factor
asymmetry
pos_neg_ratio
hf_energy
frequency
temperature
```

### Completo (16–18 features)

Adicionar:

```
period_jitter
Irms_mean_5s
Irms_std_5s
Vcc (opcional)
```

---

## 7. Normalização

Recomendado normalizar as features antes do modelo:

* Divisão por `Irms` (normalização relativa)
* Ou z-score aprendido no treinamento offline

Isso melhora estabilidade e generalização.

---

## 8. Modelo de Machine Learning

### Recomendado (treinamento offline)

* Gradient Boosting:

  * XGBoost
  * LightGBM
  * CatBoost

Motivos:

* Excelente para dados tabulares
* Aprende não linearidades complexas
* Funciona bem com poucos dados (1k–5k amostras)

### Para embarcar no ESP32

* Regressão polinomial regularizada
* MLP pequeno (2 camadas, 8–16 neurônios)
* Ou aproximação via LUT + correção

---

## 9. Coleta de Dados

Boas práticas:

* Sincronizar leituras ESP32 ↔ alicate amperímetro
* Cobrir todo o range de corrente
* Incluir diferentes tipos de carga
* Capturar transientes (liga/desliga)

Quantidade recomendada:

* **1.000 a 5.000 pares** já produzem ótimos resultados

---

## 10. Arquitetura Final

### Fase Offline

1. Coleta do sinal
2. Extração de features
3. Treinamento do modelo
4. Avaliação (MAE, erro percentual por faixa)

### Fase Embarcada

1. ESP32 calcula features
2. Aplica modelo calibrador
3. Retorna `I_corrigida ≈ I_alicate`

---

## 11. Conclusão

A abordagem de **Machine Learning para calibração** é tecnicamente sólida e superior a métodos clássicos quando:

* Há cargas não lineares
* O erro não é constante
* O sistema sofre influência de contexto físico

O sucesso do modelo depende muito mais da **qualidade das features** do que da complexidade do algoritmo.

---

## 12. Próximos Passos

* Implementar extração das features em C/C++ no ESP32
* Avaliar importância das features (SHAP)
* Testar modelos híbridos (físico + ML)
* Validar estabilidade a longo prazo

---

**Documento técnico — pronto para implementação e evolução.**





Use o comando `uvicorn backend:app --reload` para iniciar o servidor.
