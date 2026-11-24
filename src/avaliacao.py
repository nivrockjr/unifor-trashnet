"""Avaliação no conjunto de teste e análise dos erros do modelo escolhido."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    CLASSES,
    EXPERIMENTOS,
    EXPERIMENTOS_DIR,
    METRICAS,
    input_shape,
)
from src.dados import caminhos_e_rotulos, carregar_arrays, carregar_uint8, normalizar


def prever(
    modelo: keras.Model,
    caminhos: list[str],
    rotulos: np.ndarray,
    resolucao: str,
    normalizacao: str = "centralizada",
):
    altura, largura, _ = input_shape(resolucao)
    imagens, rotulos = carregar_arrays(caminhos, rotulos, altura, largura, normalizacao)
    probabilidades = modelo.predict(imagens, batch_size=BATCH_SIZE, verbose=0)
    previstos = np.argmax(probabilidades, axis=1).astype(np.int32)
    return previstos, probabilidades


def acuracia_em_partes(
    modelo: keras.Model,
    caminhos: list[str],
    rotulos: np.ndarray,
    resolucao: str,
    normalizacao: str = "centralizada",
    parte: int = 16 * 16,
) -> float:
    """Acurácia num conjunto grande, sem normalizar todas as imagens de uma vez.

    O treino em 512×384 normalizado em float32 ocuparia cerca de 4 GB. Aqui as
    imagens ficam em uint8 e só um pedaço por vez é normalizado. O pedaço é
    múltiplo do lote, então os lotes da previsão são os mesmos de uma previsão única.
    """
    altura, largura, _ = input_shape(resolucao)
    imagens = carregar_uint8(caminhos, altura, largura)
    previstos = []
    for inicio in range(0, len(imagens), parte):
        pedaco = normalizar(imagens[inicio : inicio + parte], normalizacao)
        probabilidades = modelo.predict(pedaco, batch_size=BATCH_SIZE, verbose=0)
        previstos.append(np.argmax(probabilidades, axis=1))
    return float(accuracy_score(np.asarray(rotulos), np.concatenate(previstos)))


def classes_sem_previsao(previstos: np.ndarray) -> list[str]:
    contagem = np.bincount(np.asarray(previstos), minlength=len(CLASSES))
    return [classe for classe, total in zip(CLASSES, contagem) if total == 0]


def metricas_de(rotulos: np.ndarray, previstos: np.ndarray) -> dict:
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        rotulos,
        previstos,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    por_classe = []
    for i, classe in enumerate(CLASSES):
        por_classe.append(
            {
                "classe": classe,
                "precisao": float(precisao[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "suporte": int(suporte[i]),
            }
        )
    matriz = confusion_matrix(rotulos, previstos, labels=list(range(len(CLASSES))))
    return {
        "acuracia": float(accuracy_score(rotulos, previstos)),
        "f1_macro": float(f1_score(rotulos, previstos, average="macro", zero_division=0)),
        "por_classe": por_classe,
        "matriz": matriz.tolist(),
        "relatorio": classification_report(
            rotulos,
            previstos,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        ),
    }


def carregar_registro(experimento_id: str) -> dict:
    caminho = EXPERIMENTOS_DIR / experimento_id / "registro.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def carregar_modelo(experimento_id: str) -> keras.Model:
    return keras.models.load_model(EXPERIMENTOS_DIR / experimento_id / "modelo.keras")


def avaliar_todos(raiz_dataset: Path, divisao: dict) -> dict:
    treino_x, treino_y = caminhos_e_rotulos(raiz_dataset, divisao["partes"]["treino"])
    val_x, val_y = caminhos_e_rotulos(raiz_dataset, divisao["partes"]["validacao"])
    teste_x, teste_y = caminhos_e_rotulos(raiz_dataset, divisao["partes"]["teste"])

    experimentos = []
    previsoes = {"y_val": val_y, "y_teste": teste_y}
    for espec in EXPERIMENTOS:
        registro = carregar_registro(espec["id"])
        modelo = carregar_modelo(espec["id"])
        prev_val, proba_val = prever(modelo, val_x, val_y, espec["resolucao"])
        prev_teste, proba_teste = prever(modelo, teste_x, teste_y, espec["resolucao"])
        metricas_val = metricas_de(val_y, prev_val)
        metricas_teste = metricas_de(teste_y, prev_teste)
        acuracia_treino = acuracia_em_partes(modelo, treino_x, treino_y, espec["resolucao"])
        experimentos.append(
            {
                "id": espec["id"],
                "arquitetura": espec["arquitetura"],
                "resolucao": espec["resolucao"],
                "pesos_classe": espec["pesos_classe"],
                "validacao": {k: v for k, v in metricas_val.items() if k != "relatorio"},
                "teste": {k: v for k, v in metricas_teste.items() if k != "relatorio"},
                "relatorio_teste": metricas_teste["relatorio"],
                "acuracia_treino": acuracia_treino,
                "classes_sem_previsao": {
                    "validacao": classes_sem_previsao(prev_val),
                    "teste": classes_sem_previsao(prev_teste),
                },
                "parametros": registro["parametros"],
                "tamanho_modelo_bytes": registro["tamanho_modelo_bytes"],
                "epocas_executadas": registro["epocas_executadas"],
                "epocas_maximas": registro["epocas_maximas"],
                "parou_pela_paciencia": registro["parou_pela_paciencia"],
                "epoca_modelo_salvo": registro["epoca_modelo_salvo"],
                "tempo_total_s": registro["tempo_total_s"],
                "tempo_medio_epoca_s": registro["tempo_medio_epoca_s"],
                "pico_memoria_bytes": registro["pico_memoria_bytes"],
                "medicao_memoria": registro["medicao_memoria"],
                "lote": registro["lote"],
                "hardware": registro["hardware"],
            }
        )
        previsoes[f"{espec['id']}_val"] = prev_val
        previsoes[f"{espec['id']}_teste"] = prev_teste
        previsoes[f"{espec['id']}_proba_teste"] = proba_teste
        previsoes[f"{espec['id']}_proba_val"] = proba_val
        del modelo

    escolhido = max(experimentos, key=lambda item: item["validacao"]["f1_macro"])
    # Caminho relativo à pasta do dataset, no mesmo formato do divisao.json.
    teste_relativos = [item["caminho"] for item in divisao["partes"]["teste"]]
    erros = analisar_erros(
        teste_y,
        previsoes[f"{escolhido['id']}_teste"],
        previsoes[f"{escolhido['id']}_proba_teste"],
        teste_relativos,
        escolhido["id"],
    )

    resultado = {
        "criterio_selecao": "Maior F1 macro no conjunto de validação. O teste não entra na escolha.",
        "modelo_escolhido": escolhido["id"],
        "f1_macro_validacao_escolhido": escolhido["validacao"]["f1_macro"],
        "experimentos": experimentos,
        "erros": erros,
        "zero_division": (
            "precision_recall_fscore_support e f1_score foram chamados com zero_division=0. "
            "Se uma classe não receber nenhuma previsão, precisão e F1 dessa classe ficam em 0 "
            "em vez de indefinidos."
        ),
        "zero_division_observado": _zero_division_observado(experimentos),
        "acuracia_treino_metodo": (
            "Acurácia do modelo salvo nas imagens de treino, com o dropout desligado (modo de previsão). "
            "A acurácia de treino do histórico é outra medida: a média ao longo da época, com dropout ligado."
        ),
    }

    METRICAS.mkdir(parents=True, exist_ok=True)
    (METRICAS / "resumo.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(METRICAS / "previsoes.npz", **{k: np.asarray(v) for k, v in previsoes.items()})
    salvar_tabelas(resultado)
    return resultado


def _zero_division_observado(experimentos: list[dict]) -> str:
    casos = []
    for item in experimentos:
        for conjunto, nome in (("validacao", "validação"), ("teste", "teste")):
            classes = item["classes_sem_previsao"][conjunto]
            if classes:
                casos.append(f"{item['id']} não previu {', '.join(classes)} em nenhuma imagem de {nome}")
    if not casos:
        return "Todos os modelos previram cada classe ao menos uma vez na validação e no teste."
    return "; ".join(casos) + ". Nesses casos a precisão e o F1 da classe ficaram em 0."


def analisar_erros(y_true, y_pred, proba, caminhos: list[str], experimento_id: str) -> dict:
    indices = np.where(y_true != y_pred)[0]
    ordenados = sorted(indices, key=lambda i: float(proba[i, y_pred[i]]), reverse=True)
    selecionados = ordenados[:6]
    amostras = []
    for indice in selecionados:
        real = int(y_true[indice])
        previsto = int(y_pred[indice])
        amostras.append(
            {
                "caminho": caminhos[indice],
                "classe_real": CLASSES[real],
                "classe_prevista": CLASSES[previsto],
                "probabilidade_prevista": float(proba[indice, previsto]),
            }
        )
    return {
        "experimento": experimento_id,
        "total_erros_teste": int(len(indices)),
        "total_teste": int(len(y_true)),
        "amostras": amostras,
    }


def salvar_tabelas(resultado: dict) -> None:
    linhas = []
    por_classe = []
    for item in resultado["experimentos"]:
        linhas.append(
            {
                "experimento": item["id"],
                "arquitetura": item["arquitetura"],
                "resolucao": item["resolucao"],
                "pesos_classe": item["pesos_classe"],
                "acuracia_teste": item["teste"]["acuracia"],
                "f1_macro_teste": item["teste"]["f1_macro"],
                "f1_macro_validacao": item["validacao"]["f1_macro"],
                "acuracia_validacao": item["validacao"]["acuracia"],
                "acuracia_treino": item["acuracia_treino"],
                "parametros": item["parametros"],
                "tamanho_modelo_mb": item["tamanho_modelo_bytes"] / (1024 * 1024),
                "epocas_executadas": item["epocas_executadas"],
                "parou_pela_paciencia": item["parou_pela_paciencia"],
                "epoca_modelo_salvo": item["epoca_modelo_salvo"],
                "tempo_total_min": item["tempo_total_s"] / 60,
                "tempo_medio_epoca_s": item["tempo_medio_epoca_s"],
                "pico_memoria_mb": item["pico_memoria_bytes"] / (1024 * 1024) if item["pico_memoria_bytes"] else None,
                "lote": item["lote"],
            }
        )
        for classe in item["teste"]["por_classe"]:
            por_classe.append({"experimento": item["id"], "conjunto": "teste", **classe})
            origem = next(c for c in item["validacao"]["por_classe"] if c["classe"] == classe["classe"])
            por_classe.append({"experimento": item["id"], "conjunto": "validacao", **origem})

    pd.DataFrame(linhas).to_csv(METRICAS / "experimentos.csv", index=False)
    pd.DataFrame(por_classe).drop_duplicates().to_csv(METRICAS / "por_classe.csv", index=False)


def carregar_resumo() -> dict:
    return json.loads((METRICAS / "resumo.json").read_text(encoding="utf-8"))


def miniatura(caminho: str, largura: int, altura: int) -> np.ndarray:
    with Image.open(caminho) as imagem:
        rgb = imagem.convert("RGB")
        rgb = rgb.resize((largura, altura), Image.Resampling.BILINEAR)
        return np.asarray(rgb)
