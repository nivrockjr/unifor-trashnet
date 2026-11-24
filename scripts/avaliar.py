"""Avalia os modelos salvos, sem novo treinamento, e gera figuras e tabelas."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS
from src.avaliacao import avaliar_todos
from src.dados import carregar_json, extrair_zip_se_necessario
from src.figuras import gerar_todas


def main() -> None:
    raiz = extrair_zip_se_necessario()
    eda = carregar_json(ARTIFACTS / "eda.json")
    divisao = carregar_json(ARTIFACTS / "divisao.json")
    partes = divisao["partes"]
    # Mesma ordem de listar_imagens (classe, depois nome do arquivo), para as figuras
    # de exemplos saírem iguais às do preparar_dados.py.
    eda["registros_validos"] = sorted(
        partes["treino"] + partes["validacao"] + partes["teste"],
        key=lambda item: (item["rotulo"], item["caminho"]),
    )
    resultado = avaliar_todos(raiz, divisao)
    gerar_todas(raiz, eda, divisao, resultado)
    print(f"Modelo escolhido pela validação: {resultado['modelo_escolhido']}")
    for item in resultado["experimentos"]:
        print(
            f"{item['id']:20} "
            f"F1 val {item['validacao']['f1_macro']:.3f}  "
            f"F1 teste {item['teste']['f1_macro']:.3f}  "
            f"acc teste {item['teste']['acuracia']:.3f}"
        )


if __name__ == "__main__":
    main()
