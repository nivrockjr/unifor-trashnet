# Classificação de resíduos com o TrashNet

Estudo comparativo entre MLP e CNN para a atividade final de Redes Neurais Artificiais (UNIFOR). O dataset é o TrashNet (`dataset-resized.zip`), com seis classes de resíduos. Os experimentos controlados comparam arquitetura, pesos de classe e resolução de entrada.

Repositório: https://github.com/nivrockjr/unifor-trashnet

## Equipe

| Nome | Matrícula |
| --- | --- |
| Elayne Ellen Teotonio Barbosa | 2523541 |
| Felipe Álex Gomes Nobre | 2523588 |
| Gadely Marino Camacho | 2523611 |
| Jéssica Andreia de Moura Bezerra | 2523555 |
| Nivaldo Rocha Lima | 2513690 |
| Roberto Kelsen Campos Cordeiro | 2526332 |

## O que o repositório contém

| Pasta ou arquivo | Função |
| --- | --- |
| `src/` | Preparação dos dados, modelos, treino, avaliação e figuras |
| `scripts/baixar_dataset.py` | Download e extração do ZIP oficial |
| `scripts/preparar_dados.py` | Análise exploratória e divisão estratificada |
| `scripts/treinar.py` | Seis treinamentos controlados |
| `scripts/avaliar.py` | Métricas, tabelas e figuras, sem treinar de novo |
| `scripts/ensaio_normalizacao.py` | Ensaio auxiliar: MLP com pixels só divididos por 255, comparada na validação |
| `notebooks/01_analise_exploratoria.ipynb` | Inspeção, distribuição, exemplos e tratamentos |
| `notebooks/02_treinamento.ipynb` | Configuração e execução dos seis treinos |
| `notebooks/03_resultados.ipynb` | Métricas, figuras e leitura dos resultados |
| `artifacts/` | Divisão, históricos, modelos, métricas, figuras e o ensaio auxiliar (`artifacts/ensaios/`) |

Os modelos ficam em `artifacts/experimentos/<id>/modelo.keras`. Os dois da MLP têm 54 MB cada, por causa dos 4,7 milhões de parâmetros, e os da CNN têm 1,2 MB. O registro de cada treino, com semente, histórico, época salva e tempos, fica no `registro.json` da mesma pasta.

## Dataset

Use somente o arquivo `dataset-resized.zip` do repositório [TrashNet](https://github.com/garythung/trashnet), com 2.527 imagens em 512 × 384 pixels. Não misture as imagens originais do visualizador do Hugging Face, para evitar duplicação.

Classes, nesta ordem nas matrizes: glass, paper, cardboard, plastic, metal, trash.

O próprio ZIP tem três pares de fotos idênticas com rótulos diferentes (glass115 e metal91, glass176 e plastic152, glass389 e plastic332). O notebook 01 mostra os pares. Eles foram mantidos, e nenhum está no teste.

Citação do dataset: Thung, G.; Yang, M. *TrashNet*. 2017. Disponível em: https://github.com/garythung/trashnet.

## Ambiente

Os resultados deste repositório foram gerados num Mac com processador Apple M2 (8 núcleos) e 8 GB de RAM, com macOS 27.0 e Python 3.12.4. O TensorFlow não encontrou GPU, então treino e avaliação rodaram só na CPU. Os seis treinos rodaram em 17/11/2026, um de cada vez, das 12:38 às 14:04.

Na mesma máquina, o treino é determinístico: duas execuções completas dos seis experimentos geraram modelos com pesos idênticos. Em outra máquina, os números podem variar um pouco, porque as contas em ponto flutuante mudam com o processador. Para conferir, refaça um treino com `--substituir`, rode `python scripts/avaliar.py` e veja que o `git status` não acusa mudança em `artifacts/metricas/previsoes.npz`, que guarda as previsões dos seis modelos.

As versões estão fixadas em `requirements.txt`, para Python 3.12: TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1, Matplotlib 3.11.2, Pillow 13.0.0 e psutil 7.2.2.

## Instalação

Requer Python 3.12. Na raiz do projeto:

macOS ou Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No Windows, a instalação do TensorFlow pode falhar se o caminho da pasta do projeto for muito longo. Nesse caso, clone o projeto numa pasta curta, como `C:\trashnet`, ou ative o suporte a caminhos longos do sistema.

Os comandos abaixo são os mesmos nos três sistemas, com o ambiente ativado.

## Execução

A divisão usa a semente 42 e é gravada em `artifacts/divisao.json`. Os treinos seguintes leem esse arquivo, então a separação não muda entre experimentos.

```bash
python scripts/baixar_dataset.py
python scripts/preparar_dados.py
python scripts/treinar.py
python scripts/ensaio_normalizacao.py
python scripts/avaliar.py
```

`treinar.py` pula um experimento que já tenha `modelo.keras` e `registro.json`. Para refazer um treino:

```bash
python scripts/treinar.py --id cnn_128x96 --substituir
```

Os notebooks repetem esses passos. O notebook de resultados também pode ser aberto depois do `avaliar.py`. Para executar um notebook pela linha de comando e gravar as saídas no próprio arquivo:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/03_resultados.ipynb
```

## Avaliação sem novo treinamento

Com os seis modelos e a divisão já salvos, e com o dataset extraído em `data/`:

```bash
python scripts/avaliar.py
```

O script carrega os modelos, calcula acurácia, F1 macro e métricas por classe na validação e no teste, e a acurácia no treino com o modelo salvo e o dropout desligado. Ele grava `artifacts/metricas/` e regenera as figuras em `artifacts/figuras/`. O conjunto de teste não é usado para escolher o modelo. A escolha usa o maior F1 macro de validação.

## Protocolo dos experimentos

| Id | Modelo | Entrada (largura × altura) | Pesos de classe |
| --- | --- | --- | --- |
| `mlp_128x96` | MLP | 128 × 96 | não |
| `mlp_128x96_pesos` | MLP | 128 × 96 | sim |
| `cnn_128x96` | CNN | 128 × 96 | não |
| `cnn_128x96_pesos` | CNN | 128 × 96 | sim |
| `cnn_256x192` | CNN | 256 × 192 | não |
| `cnn_512x384` | CNN | 512 × 384 | não |

- Divisão estratificada de 70% / 15% / 15%, semente 42, a mesma em todos os experimentos.
- Conversão para RGB, redimensionamento bilinear a partir da imagem de referência e normalização fixa `(pixel / 255) - 0,5`. Não há média nem desvio calculados nos dados.
- Sem aumento de dados nos treinos controlados. Nenhuma imagem de validação ou teste entra no treino.
- Perda: entropia cruzada categórica esparsa. Otimizador: Adam com taxa 0,001. Lote 16 nos seis experimentos, inclusive em 512 × 384.
- As imagens de treino e validação são carregadas e redimensionadas uma vez, em memória, do mesmo jeito nas três resoluções.
- Até 60 épocas, com parada antecipada pela perda de validação (paciência 5, melhora mínima 0,0001). O arquivo salvo é o da menor perda de validação. O `registro.json` indica se o treino parou pela paciência ou pelo teto.
- Pesos de classe, quando usados, são `balanced` do scikit-learn, calculados só no treino.
- A CNN usa três blocos convolução + max pooling e *global average pooling*, para manter a quantidade de parâmetros nas três resoluções.
- A MLP vetoriza os pixels e usa duas camadas ocultas com ReLU e dropout.

### Medição de custo

- O tempo de cada época é medido entre o início e o fim da época, incluindo a validação. Ficam de fora o download, a divisão e a carga das imagens, registrada à parte em `tempo_carga_imagens_s`.
- Cada experimento roda num processo Python próprio: sem `--id`, `treinar.py` chama a si mesmo com `--id` para cada experimento, um de cada vez. Assim o pico de memória de um treino não se mistura com o de outro.
- O pico de memória é o que o sistema operacional informa para esse processo. No macOS, o pico do *memory footprint* (`ri_lifetime_max_phys_footprint`), que é o valor da coluna Memória do Monitor de Atividade e conta também a memória comprimida ou enviada ao swap. No Linux, `ru_maxrss`. No Windows, `peak_wset`. O valor inclui o TensorFlow carregado e as imagens de treino e validação em memória.
- Cada `registro.json` guarda o horário de início e de fim, a CPU usada por outros programas durante o treino (amostrada a cada 5 s) e a RAM livre e o swap no começo e no fim, para mostrar que o treino rodou sozinho.
- `treinar.py` cria a trava `artifacts/.treino_em_andamento` e se recusa a começar se outro treino estiver rodando, porque dois processos dividindo a CPU distorcem o tempo.

Não houve GPU, então não há pico de memória de GPU. Os números de custo de cada treino ficam no `registro.json` da pasta do experimento e, reunidos, em `artifacts/metricas/experimentos.csv` (parâmetros, tamanho do modelo, épocas, tempo total, tempo médio por época, pico de memória e lote). A saída completa da execução fica em `artifacts/treino.log`.

## Ensaio auxiliar de normalização

`scripts/ensaio_normalizacao.py` treina a MLP 128 × 96 sem pesos com os pixels só divididos por 255 (sugestão do enunciado) e compara com a normalização dos seis experimentos, apenas na validação. O resultado fica em `artifacts/ensaios/ensaio_normalizacao.json`. Ele não faz parte dos seis experimentos e não usa o conjunto de teste. O modelo desse ensaio (54 MB) não vai para o GitHub. Se ele não existir, o script treina de novo, em cerca de 30 segundos.

```bash
python scripts/ensaio_normalizacao.py
```
