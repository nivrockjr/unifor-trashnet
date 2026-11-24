"""Constantes compartilhadas pelos experimentos controlados."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "dataset-resized.zip"
DATASET_DIRNAME = "dataset-resized"

ARTIFACTS = ROOT / "artifacts"
FIGURAS = ARTIFACTS / "figuras"
METRICAS = ARTIFACTS / "metricas"
EXPERIMENTOS_DIR = ARTIFACTS / "experimentos"
ENSAIOS_DIR = ARTIFACTS / "ensaios"
NOTEBOOKS = ROOT / "notebooks"
TRAVA_TREINO = ARTIFACTS / ".treino_em_andamento"

# Ordem exigida nas matrizes de confusão.
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
CLASSES_PT = {
    "glass": "vidro",
    "paper": "papel",
    "cardboard": "papelão",
    "plastic": "plástico",
    "metal": "metal",
    "trash": "outros resíduos",
}

SEED = 42
# Frações aproximadas: 70% treino e 30% reservados, divididos ao meio em validação e teste.
TEST_SIZE_TEMP = 0.30
TEST_SIZE_SPLIT = 0.50

# Resolução informada como largura x altura. O Keras recebe altura x largura x canais.
RESOLUCOES = {
    "128x96": {"largura": 128, "altura": 96},
    "256x192": {"largura": 256, "altura": 192},
    "512x384": {"largura": 512, "altura": 384},
}

# Mesmo lote nos seis treinos, para não confundir o custo da resolução com o tamanho do lote.
BATCH_SIZE = 16
# Teto alto o bastante para a parada antecipada decidir o fim do treino em todos os experimentos.
EPOCHS = 60
PACIENCIA = 5
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
OTIMIZADOR = "adam"
PERDA = "sparse_categorical_crossentropy"

DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DATASET_URL_ALTERNATIVA = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
)
# SHA-256 dos dois ZIPs conferidos em 17/11/2026. Os dois têm as mesmas 2.527 imagens
# (mesmos nomes e mesmo CRC de cada arquivo); só o empacotamento do ZIP difere.
ZIP_SHA256_ACEITOS = {
    "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2": "GitHub garythung/trashnet",
    "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c": "espelho no Hugging Face",
}
# Tabela de imagens por classe do enunciado.
CONTAGEM_ESPERADA = {
    "glass": 501,
    "paper": 594,
    "cardboard": 403,
    "plastic": 482,
    "metal": 410,
    "trash": 137,
}

EXPERIMENTOS = [
    {
        "id": "mlp_128x96",
        "arquitetura": "mlp",
        "resolucao": "128x96",
        "pesos_classe": False,
    },
    {
        "id": "mlp_128x96_pesos",
        "arquitetura": "mlp",
        "resolucao": "128x96",
        "pesos_classe": True,
    },
    {
        "id": "cnn_128x96",
        "arquitetura": "cnn",
        "resolucao": "128x96",
        "pesos_classe": False,
    },
    {
        "id": "cnn_128x96_pesos",
        "arquitetura": "cnn",
        "resolucao": "128x96",
        "pesos_classe": True,
    },
    {
        "id": "cnn_256x192",
        "arquitetura": "cnn",
        "resolucao": "256x192",
        "pesos_classe": False,
    },
    {
        "id": "cnn_512x384",
        "arquitetura": "cnn",
        "resolucao": "512x384",
        "pesos_classe": False,
    },
]


# Ensaio auxiliar, fora dos seis experimentos: a MLP sem pesos com a normalização
# sugerida no enunciado (só dividir por 255). Avaliado apenas na validação.
ENSAIO_NORMALIZACAO = {
    "id": "mlp_128x96_div255",
    "arquitetura": "mlp",
    "resolucao": "128x96",
    "pesos_classe": False,
    "normalizacao": "div255",
}

# Normalizações aceitas. A dos seis experimentos é "centralizada".
NORMALIZACOES = {
    "centralizada": "Pixels RGB / 255, depois subtração de 0,5. Transformação fixa, igual em treino, validação e teste.",
    "div255": "Pixels RGB / 255, sem subtração. Transformação fixa, igual em treino e validação.",
}


def input_shape(resolucao: str) -> tuple[int, int, int]:
    spec = RESOLUCOES[resolucao]
    return (spec["altura"], spec["largura"], 3)


def nome_exibicao(experimento_id: str) -> str:
    mapa = {
        "mlp_128x96": "MLP 128×96",
        "mlp_128x96_pesos": "MLP 128×96 + pesos",
        "cnn_128x96": "CNN 128×96",
        "cnn_128x96_pesos": "CNN 128×96 + pesos",
        "cnn_256x192": "CNN 256×192",
        "cnn_512x384": "CNN 512×384",
        "mlp_128x96_div255": "MLP 128×96, só /255",
    }
    return mapa[experimento_id]
