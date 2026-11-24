"""Inspeção do TrashNet e divisão estratificada 70/15/15."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS, SEED
from src.dados import (
    caminho_relativo,
    dividir_estratificado,
    extrair_zip_se_necessario,
    inspecionar_imagens,
    listar_imagens,
    salvar_json,
)
from src.figuras import figura_distribuicao, figura_exemplos, figura_resolucoes


def main() -> None:
    raiz = extrair_zip_se_necessario()
    registros = listar_imagens(raiz)
    eda = inspecionar_imagens(raiz, registros)
    validos = eda["registros_validos"]
    divisao = dividir_estratificado(validos)

    eda_publica = {chave: valor for chave, valor in eda.items() if chave != "registros_validos"}
    eda_publica["raiz_dataset"] = caminho_relativo(raiz)
    eda_publica["semente"] = SEED
    eda_publica["observacoes"] = [
        "Imagens com falha de leitura são excluídas antes da divisão.",
        "Todas as imagens deste ZIP já estão em RGB JPEG 512×384.",
        "A divisão é estratificada e reutilizada por todos os experimentos.",
        "Normalização fixa: pixel/255 - 0,5, sem estatísticas calculadas nos dados.",
        "Não há aumento de dados nos experimentos controlados, para isolar arquitetura, pesos e resolução.",
    ]
    salvar_json(ARTIFACTS / "eda.json", eda_publica)
    salvar_json(
        ARTIFACTS / "divisao.json",
        {
            "semente": SEED,
            "fracoes_alvo": {"treino": 0.70, "validacao": 0.15, "teste": 0.15},
            "metodo": (
                "train_test_split com test_size=0.30 estratificado e, em seguida, "
                "divisão ao meio do restante, também estratificada. Semente 42."
            ),
            "raiz_dataset": caminho_relativo(raiz),
            "contagens": divisao["contagens"],
            "partes": divisao["partes"],
        },
    )
    figura_distribuicao(divisao["contagens"])
    figura_exemplos(raiz, validos)
    figura_resolucoes(raiz, validos)

    print(f"Dataset: {caminho_relativo(raiz)}")
    print(f"Válidas: {eda['total_validos']} de {eda['total_arquivos']}")
    print(f"Dimensões: {eda['dimensoes']}")
    print(f"Modos: {eda['modos']}")
    print(f"Canais: {eda['canais']}")
    print(f"Formatos: {eda['formatos']}")
    print(f"Problemas: {len(eda['problemas'])}")
    for nome, contagem in divisao["contagens"].items():
        print(nome, contagem, "total", sum(contagem.values()))


if __name__ == "__main__":
    main()
