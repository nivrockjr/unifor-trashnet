"""Leitura do TrashNet, análise exploratória e divisão estratificada."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    DATASET_DIRNAME,
    DATA_DIR,
    ROOT,
    SEED,
    TEST_SIZE_SPLIT,
    TEST_SIZE_TEMP,
    ZIP_PATH,
)

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def caminho_relativo(caminho: Path) -> str:
    """Caminho a partir da raiz do projeto, para os JSON valerem em qualquer máquina."""
    try:
        return Path(caminho).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(caminho).as_posix()


def localizar_dataset() -> Path:
    """Encontra a pasta que contém as seis classes, mesmo com um nível extra no ZIP."""
    candidatos = [
        DATA_DIR / DATASET_DIRNAME,
        DATA_DIR / "dataset-resized" / "dataset-resized",
    ]
    for pasta in candidatos:
        if pasta.is_dir() and all((pasta / classe).is_dir() for classe in CLASSES):
            return pasta

    for pasta in DATA_DIR.rglob("*"):
        if pasta.is_dir() and all((pasta / classe).is_dir() for classe in CLASSES):
            return pasta

    raise FileNotFoundError(
        "Dataset não encontrado. Execute scripts/baixar_dataset.py antes da preparação."
    )


def extrair_zip_se_necessario() -> Path:
    try:
        return localizar_dataset()
    except FileNotFoundError:
        if not ZIP_PATH.is_file():
            raise
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as arquivo:
            arquivo.extractall(DATA_DIR)
        return localizar_dataset()


def listar_imagens(raiz: Path) -> list[dict]:
    registros = []
    for indice, classe in enumerate(CLASSES):
        pasta = raiz / classe
        for caminho in sorted(pasta.iterdir()):
            if caminho.suffix.lower() not in EXTENSOES or not caminho.is_file():
                continue
            relativo = caminho.relative_to(raiz).as_posix()
            registros.append(
                {
                    "caminho": relativo,
                    "classe": classe,
                    "rotulo": indice,
                }
            )
    return registros


def inspecionar_imagens(raiz: Path, registros: list[dict]) -> dict:
    """Abre cada arquivo, confere integridade e resume dimensões, modo e canais."""
    problemas = []
    modos = Counter()
    dimensoes = Counter()
    canais = Counter()
    formatos = Counter()

    for item in registros:
        caminho = raiz / item["caminho"]
        try:
            with Image.open(caminho) as imagem:
                imagem.load()
                modo = imagem.mode
                largura, altura = imagem.size
                formato = imagem.format or caminho.suffix.lower().lstrip(".")
            if modo == "RGB":
                n_canais = 3
            elif modo == "RGBA":
                n_canais = 4
            elif modo == "L":
                n_canais = 1
            else:
                n_canais = None
            modos[modo] += 1
            dimensoes[f"{largura}x{altura}"] += 1
            canais[str(n_canais)] += 1
            formatos[str(formato)] += 1
        except (UnidentifiedImageError, OSError, ValueError) as erro:
            problemas.append({"caminho": item["caminho"], "erro": str(erro)})

    validos = [item for item in registros if item["caminho"] not in {p["caminho"] for p in problemas}]
    contagem = {classe: sum(item["classe"] == classe for item in validos) for classe in CLASSES}

    return {
        "total_arquivos": len(registros),
        "total_validos": len(validos),
        "problemas": problemas,
        "modos": dict(modos),
        "dimensoes": dict(dimensoes),
        "canais": dict(canais),
        "formatos": dict(formatos),
        "contagem_classes": contagem,
        "registros_validos": validos,
    }



def duplicatas_por_conteudo(raiz: Path, divisao: dict) -> list[list[dict]]:
    """Grupos de arquivos com o mesmo conteúdo (SHA-256 igual), com a classe e o conjunto de cada um."""
    grupos: dict[str, list[dict]] = {}
    for conjunto, itens in divisao["partes"].items():
        for item in itens:
            resumo = hashlib.sha256((raiz / item["caminho"]).read_bytes()).hexdigest()
            grupos.setdefault(resumo, []).append(
                {"arquivo": Path(item["caminho"]).name, "classe": CLASSES[item["rotulo"]], "conjunto": conjunto}
            )
    return [grupo for grupo in grupos.values() if len(grupo) > 1]

def dividir_estratificado(registros: list[dict]) -> dict:
    """70/15/15 estratificado. A mesma imagem permanece em um único conjunto."""
    indices = np.arange(len(registros))
    rotulos = np.array([item["rotulo"] for item in registros])

    treino_idx, temp_idx = train_test_split(
        indices,
        test_size=TEST_SIZE_TEMP,
        stratify=rotulos,
        random_state=SEED,
    )
    val_idx, teste_idx = train_test_split(
        temp_idx,
        test_size=TEST_SIZE_SPLIT,
        stratify=rotulos[temp_idx],
        random_state=SEED,
    )

    def selecionar(idxs: np.ndarray) -> list[dict]:
        return [registros[int(i)] for i in sorted(idxs)]

    partes = {
        "treino": selecionar(treino_idx),
        "validacao": selecionar(val_idx),
        "teste": selecionar(teste_idx),
    }
    contagens = {
        nome: {classe: sum(item["classe"] == classe for item in itens) for classe in CLASSES}
        for nome, itens in partes.items()
    }
    return {"partes": partes, "contagens": contagens}


def salvar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def caminhos_e_rotulos(raiz: Path, itens: list[dict]) -> tuple[list[str], np.ndarray]:
    caminhos = [str(raiz / item["caminho"]) for item in itens]
    rotulos = np.array([item["rotulo"] for item in itens], dtype=np.int32)
    return caminhos, rotulos


def _ler_rgb(caminho: str, altura: int, largura: int) -> np.ndarray:
    with Image.open(caminho) as imagem:
        rgb = imagem.convert("RGB").resize((largura, altura), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def carregar_uint8(caminhos: list[str], altura: int, largura: int) -> np.ndarray:
    """RGB redimensionado, ainda de 0 a 255, para caber na memória nas resoluções maiores."""
    imagens = np.empty((len(caminhos), altura, largura, 3), dtype=np.uint8)
    for i, caminho in enumerate(caminhos):
        imagens[i] = _ler_rgb(caminho, altura, largura)
    return imagens


def normalizar(imagens: np.ndarray, modo: str = "centralizada") -> np.ndarray:
    """Normalização fixa dos pixels, sem estatística calculada nos dados.

    "centralizada": divide por 255 e subtrai 0,5, levando a [-0,5, 0,5]. É a dos
    seis experimentos. "div255": só divide por 255, levando a [0, 1]. É a do
    ensaio auxiliar (scripts/ensaio_normalizacao.py), que compara as duas.
    Como nada é estimado no treino, não há vazamento para validação e teste.
    """
    escalado = imagens.astype(np.float32) / 255.0
    if modo == "centralizada":
        return escalado - 0.5
    if modo == "div255":
        return escalado
    raise ValueError(f"Normalização desconhecida: {modo}")


def carregar_arrays(
    caminhos: list[str],
    rotulos: np.ndarray,
    altura: int,
    largura: int,
    normalizacao: str = "centralizada",
) -> tuple[np.ndarray, np.ndarray]:
    """Converte para RGB, redimensiona e aplica a normalização fixa."""
    imagens = normalizar(carregar_uint8(caminhos, altura, largura), normalizacao)
    return imagens, rotulos
