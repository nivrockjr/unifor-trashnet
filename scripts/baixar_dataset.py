"""Baixa, confere e extrai o dataset-resized.zip do TrashNet."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CLASSES,
    CONTAGEM_ESPERADA,
    DATASET_URL,
    DATASET_URL_ALTERNATIVA,
    ZIP_PATH,
    ZIP_SHA256_ACEITOS,
)
from src.dados import EXTENSOES, extrair_zip_se_necessario, localizar_dataset

# O Python do instalador python.org no macOS não usa os certificados do sistema.
# O pacote certifi traz a lista de autoridades, e o download funciona em qualquer sistema.
CONTEXTO_SSL = ssl.create_default_context(cafile=certifi.where())


def baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    print(f"Baixando {url}")
    with urllib.request.urlopen(url, context=CONTEXTO_SSL, timeout=120) as resposta, parcial.open("wb") as arquivo:
        shutil.copyfileobj(resposta, arquivo)
    parcial.replace(destino)
    print(f"Arquivo salvo em {destino} ({destino.stat().st_size / 1024**2:.1f} MB)")


def sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir_zip(caminho: Path) -> None:
    valor = sha256(caminho)
    if valor not in ZIP_SHA256_ACEITOS:
        raise SystemExit(
            f"O ZIP em {caminho} não é o dataset-resized.zip indicado (SHA-256 {valor}). "
            "Apague o arquivo e rode o script de novo."
        )
    print(f"ZIP conferido: {ZIP_SHA256_ACEITOS[valor]} (SHA-256 {valor[:12]}...)")


def conferir_contagem(pasta: Path) -> None:
    contagem = {
        classe: sum(1 for item in (pasta / classe).iterdir() if item.suffix.lower() in EXTENSOES)
        for classe in CLASSES
    }
    if contagem != CONTAGEM_ESPERADA:
        raise SystemExit(f"Contagem por classe diferente da esperada: {contagem}")
    print(f"Contagem por classe conferida: {contagem}, total {sum(contagem.values())}")


def main() -> None:
    try:
        pasta = localizar_dataset()
        print(f"Dataset já disponível em {pasta}")
        conferir_contagem(pasta)
        return
    except FileNotFoundError:
        pass

    if not ZIP_PATH.is_file():
        try:
            baixar(DATASET_URL, ZIP_PATH)
        except Exception as erro:
            print(f"Fonte principal falhou ({erro}). Tentando o espelho.")
            baixar(DATASET_URL_ALTERNATIVA, ZIP_PATH)

    conferir_zip(ZIP_PATH)
    pasta = extrair_zip_se_necessario()
    conferir_contagem(pasta)
    print(f"Dataset extraído em {pasta}")


if __name__ == "__main__":
    main()
