"""Ensaio auxiliar: MLP 128×96 sem pesos com pixels só divididos por 255.

Fica fora dos seis experimentos. Compara, apenas na validação, a normalização
sugerida no enunciado (pixel / 255) com a usada nos seis treinos
((pixel / 255) - 0,5). O conjunto de teste não é usado aqui.

Rode depois de scripts/treinar.py, para existir o modelo mlp_128x96.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tensorflow import keras

from src.avaliacao import metricas_de, prever
from src.config import ARTIFACTS, CLASSES, ENSAIO_NORMALIZACAO, ENSAIOS_DIR, EXPERIMENTOS_DIR, input_shape
from src.dados import caminhos_e_rotulos, carregar_json, carregar_uint8, extrair_zip_se_necessario
from src.trava import trava_de_treino
from src.treino import treinar_um


def media_dos_pixels(caminhos: list[str], resolucao: str) -> dict:
    """Média dos pixels de treino depois de dividir por 255, geral e por canal."""
    altura, largura, _ = input_shape(resolucao)
    imagens = carregar_uint8(caminhos, altura, largura).astype(np.float64) / 255.0
    por_canal = imagens.mean(axis=(0, 1, 2))
    return {
        "geral": round(float(imagens.mean()), 4),
        "R": round(float(por_canal[0]), 4),
        "G": round(float(por_canal[1]), 4),
        "B": round(float(por_canal[2]), 4),
    }


def resumo_validacao(modelo_dir: Path, normalizacao: str, val_caminhos, val_y) -> dict:
    registro = json.loads((modelo_dir / "registro.json").read_text(encoding="utf-8"))
    modelo = keras.models.load_model(modelo_dir / "modelo.keras")
    previstos, _ = prever(modelo, val_caminhos, val_y, registro["resolucao"], normalizacao)
    metricas = metricas_de(val_y, previstos)
    contagem = np.bincount(previstos, minlength=len(CLASSES))
    return {
        "id": registro["id"],
        "normalizacao": registro["normalizacao"],
        "epocas_executadas": registro["epocas_executadas"],
        "epoca_modelo_salvo": registro["epoca_modelo_salvo"],
        "acuracia_validacao": metricas["acuracia"],
        "f1_macro_validacao": metricas["f1_macro"],
        "previsoes_por_classe": {classe: int(n) for classe, n in zip(CLASSES, contagem)},
        "classes_nunca_previstas": [classe for classe, n in zip(CLASSES, contagem) if n == 0],
        "matriz_validacao": metricas["matriz"],
        "historico_val_accuracy": registro["historico"]["val_accuracy"],
    }


def main() -> None:
    divisao = carregar_json(ARTIFACTS / "divisao.json")
    raiz = extrair_zip_se_necessario()
    base = EXPERIMENTOS_DIR / "mlp_128x96"
    if not (base / "modelo.keras").is_file():
        raise SystemExit("Rode scripts/treinar.py antes: o ensaio compara com o modelo mlp_128x96.")

    with trava_de_treino():
        treinar_um(ENSAIO_NORMALIZACAO, raiz, divisao, pasta_base=ENSAIOS_DIR)

    treino_caminhos, _ = caminhos_e_rotulos(raiz, divisao["partes"]["treino"])
    val_caminhos, val_y = caminhos_e_rotulos(raiz, divisao["partes"]["validacao"])
    resultado = {
        "objetivo": (
            "Comparar, só na validação, a MLP 128×96 sem pesos com pixel / 255 "
            "e com (pixel / 255) - 0,5. Tudo o mais é igual: divisão, semente, arquitetura, "
            "otimizador, lote e parada."
        ),
        "media_pixels_treino_div255": media_dos_pixels(treino_caminhos, ENSAIO_NORMALIZACAO["resolucao"]),
        "div255": resumo_validacao(ENSAIOS_DIR / ENSAIO_NORMALIZACAO["id"], "div255", val_caminhos, val_y),
        "centralizada": resumo_validacao(base, "centralizada", val_caminhos, val_y),
    }
    destino = ENSAIOS_DIR / "ensaio_normalizacao.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Média dos pixels de treino (/255): {resultado['media_pixels_treino_div255']}")
    for chave in ["div255", "centralizada"]:
        item = resultado[chave]
        print(
            f"{chave:12} acc val {item['acuracia_validacao']:.3f}  F1 macro val {item['f1_macro_validacao']:.3f}  "
            f"previsões por classe {item['previsoes_por_classe']}"
        )
    print(f"Gravado em {destino.relative_to(ARTIFACTS.parent)}")


if __name__ == "__main__":
    main()
