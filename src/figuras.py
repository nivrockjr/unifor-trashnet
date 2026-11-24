"""Figuras exigidas pelo enunciado.

As figuras são desenhadas na largura da coluna do artigo (16 cm), para que o
tamanho da letra no papel seja o tamanho definido aqui. O título de cada figura
fica na legenda do artigo, e não dentro da imagem. Os eixos usam vírgula decimal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend fixo: as figuras saem iguais no terminal, no notebook e em qualquer sistema.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PIL import Image

from src.config import (
    CLASSES,
    CLASSES_PT,
    EXPERIMENTOS,
    EXPERIMENTOS_DIR,
    FIGURAS,
    RESOLUCOES,
    nome_exibicao,
)
from src.dados import carregar_json

LARGURA_POL = 6.3  # 16 cm, a largura do texto no artigo
DPI = 200
COR_TREINO = "#1f77b4"
COR_VALIDACAO = "#ff7f0e"


def _estilo():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def virgula(valor: float, casas: int | None = None) -> str:
    texto = f"{valor:.{casas}f}" if casas is not None else f"{valor:g}"
    return texto.replace(".", ",")


def _eixo_y_com_virgula(eixo) -> None:
    eixo.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: virgula(valor)))


def _salvar(fig, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


def _rotulo_classe(classe: str) -> str:
    return f"{classe}\n({CLASSES_PT[classe]})"


def figura_distribuicao(contagens: dict, destino: Path | None = None) -> Path:
    _estilo()
    destino = destino or FIGURAS / "distribuicao_classes.png"
    conjuntos = ["treino", "validacao", "teste"]
    titulos = {"treino": "Treino", "validacao": "Validação", "teste": "Teste"}
    x = np.arange(len(CLASSES))
    largura = 0.27
    fig, ax = plt.subplots(figsize=(LARGURA_POL, 2.35))
    for i, nome in enumerate(conjuntos):
        valores = [contagens[nome][classe] for classe in CLASSES]
        barras = ax.bar(x + (i - 1) * largura, valores, width=largura, label=titulos[nome])
        ax.bar_label(barras, fontsize=6, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_rotulo_classe(c) for c in CLASSES])
    ax.set_ylabel("Imagens")
    ax.set_ylim(0, max(contagens["treino"].values()) * 1.15)
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return _salvar(fig, destino)


def figura_exemplos(raiz: Path, registros: list[dict], destino: Path | None = None) -> Path:
    """Duas imagens de cada classe, na ordem exigida."""
    _estilo()
    destino = destino or FIGURAS / "exemplos_classes.png"
    fig, eixos = plt.subplots(2, 6, figsize=(LARGURA_POL, 2.35))
    for coluna, classe in enumerate(CLASSES):
        exemplos = [item for item in registros if item["classe"] == classe][:2]
        for linha, item in enumerate(exemplos):
            with Image.open(raiz / item["caminho"]) as imagem:
                eixos[linha, coluna].imshow(imagem.convert("RGB"))
            eixos[linha, coluna].axis("off")
            if linha == 0:
                eixos[linha, coluna].set_title(_rotulo_classe(classe), fontsize=7.5)
    fig.tight_layout(pad=0.3)
    return _salvar(fig, destino)


def imagem_das_resolucoes(registros: list[dict]) -> dict:
    """A imagem usada na figura das três resoluções: o primeiro papel, na ordem da lista."""
    return next(item for item in registros if item["classe"] == "paper")


def figura_resolucoes(raiz: Path, registros: list[dict], destino: Path | None = None) -> Path:
    """A mesma imagem nas três resoluções de entrada."""
    _estilo()
    destino = destino or FIGURAS / "exemplo_resolucoes.png"
    papel = imagem_das_resolucoes(registros)
    with Image.open(raiz / papel["caminho"]) as imagem:
        original = imagem.convert("RGB")
        paineis = []
        for nome, spec in RESOLUCOES.items():
            redim = original.resize((spec["largura"], spec["altura"]), Image.Resampling.BILINEAR)
            paineis.append((nome, redim))
    fig, eixos = plt.subplots(1, 3, figsize=(LARGURA_POL, 1.95))
    for eixo, (_, imagem) in zip(eixos, paineis):
        eixo.imshow(imagem)
        eixo.set_title(f"{imagem.size[0]} × {imagem.size[1]} pixels")
        eixo.axis("off")
    fig.tight_layout(pad=0.3)
    return _salvar(fig, destino)


def figura_curvas(destino: Path | None = None) -> Path:
    """Perda e acurácia por época, em pares: sem e com pesos lado a lado, e as resoluções da CNN."""
    _estilo()
    destino = destino or FIGURAS / "curvas_aprendizagem.png"
    ordem = [
        ["mlp_128x96", "mlp_128x96_pesos"],
        ["cnn_128x96", "cnn_128x96_pesos"],
        ["cnn_256x192", "cnn_512x384"],
    ]
    ids_validos = {espec["id"] for espec in EXPERIMENTOS}
    assert {i for linha in ordem for i in linha} == ids_validos
    fig = plt.figure(figsize=(LARGURA_POL, 5.4))
    subfiguras = fig.subfigures(3, 2, hspace=0.04, wspace=0.03)
    for linha, pares in enumerate(ordem):
        for coluna, espec_id in enumerate(pares):
            registro = carregar_json(EXPERIMENTOS_DIR / espec_id / "registro.json")
            hist = registro["historico"]
            salva = registro["epoca_modelo_salvo"]
            epocas = np.arange(1, len(hist["loss"]) + 1)
            sub = subfiguras[linha, coluna]
            sub.suptitle(nome_exibicao(espec_id), fontsize=8.5, fontweight="bold", y=0.99)
            eixos = sub.subplots(1, 2)
            for eixo, chave, titulo in (
                (eixos[0], "loss", "Perda"),
                (eixos[1], "accuracy", "Acurácia"),
            ):
                eixo.plot(epocas, hist[chave], color=COR_TREINO, linewidth=1.1)
                eixo.plot(epocas, hist[f"val_{chave}"], color=COR_VALIDACAO, linewidth=1.1)
                eixo.axvline(salva, color="0.35", linestyle="--", linewidth=0.9)
                eixo.set_title(titulo, fontsize=7.5, pad=2)
                eixo.grid(alpha=0.3)
                eixo.tick_params(labelsize=6.5)
                _eixo_y_com_virgula(eixo)
                if linha == len(ordem) - 1:
                    eixo.set_xlabel("Época", fontsize=7)
            sub.subplots_adjust(left=0.13, right=0.98, bottom=0.2 if linha == 2 else 0.12, top=0.8, wspace=0.42)
    legenda = [
        Line2D([0], [0], color=COR_TREINO, linewidth=1.4, label="Treino"),
        Line2D([0], [0], color=COR_VALIDACAO, linewidth=1.4, label="Validação"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.0, label="Época do modelo salvo"),
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    return _salvar(fig, destino)


def figura_matrizes(resultado: dict, destino: Path | None = None) -> Path:
    _estilo()
    destino = destino or FIGURAS / "matrizes_confusao.png"
    fig, eixos = plt.subplots(2, 3, figsize=(LARGURA_POL, 4.85))
    imagem = None
    for posicao, (eixo, item) in enumerate(zip(eixos.ravel(), resultado["experimentos"])):
        matriz = np.array(item["teste"]["matriz"], dtype=float)
        soma = matriz.sum(axis=1, keepdims=True)
        percentual = np.divide(matriz, np.maximum(soma, 1.0), out=np.zeros_like(matriz), where=soma > 0) * 100
        imagem = eixo.imshow(percentual, cmap="Blues", vmin=0, vmax=100)
        eixo.set_title(nome_exibicao(item["id"]), fontsize=7.5, fontweight="bold", pad=3)
        eixo.set_xticks(range(len(CLASSES)))
        eixo.set_yticks(range(len(CLASSES)))
        eixo.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=6.3)
        eixo.set_yticklabels(CLASSES if posicao % 3 == 0 else [], fontsize=6.3)
        eixo.tick_params(length=2, pad=1.5)
        if posicao // 3 == 1:
            eixo.set_xlabel("Classe prevista", fontsize=7)
        if posicao % 3 == 0:
            eixo.set_ylabel("Classe real", fontsize=7)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                texto = f"{int(matriz[i, j])}\n{percentual[i, j]:.0f}%"
                cor = "white" if percentual[i, j] > 70 else "black"
                eixo.text(j, i, texto, ha="center", va="center", fontsize=5.6, color=cor, linespacing=1.05)
    fig.tight_layout(rect=(0, 0, 0.93, 1), h_pad=0.8, w_pad=0.4)
    barra = fig.colorbar(imagem, ax=eixos.ravel().tolist(), fraction=0.02, pad=0.015)
    barra.set_label("% da linha (classe real)", fontsize=6.5)
    barra.ax.tick_params(labelsize=6)
    return _salvar(fig, destino)


def figura_efeito_resolucao(resultado: dict, destino: Path | None = None) -> Path:
    """CNN sem pesos nas três resoluções: desempenho no teste, memória e tempo, em quatro painéis."""
    _estilo()
    destino = destino or FIGURAS / "efeito_resolucao.png"
    ids = ["cnn_128x96", "cnn_256x192", "cnn_512x384"]
    por_id = {item["id"]: item for item in resultado["experimentos"]}
    nomes = ["128 × 96", "256 × 192", "512 × 384"]
    acuracia = [por_id[i]["teste"]["acuracia"] for i in ids]
    f1 = [por_id[i]["teste"]["f1_macro"] for i in ids]
    memoria = [(por_id[i]["pico_memoria_bytes"] or 0) / 1024**2 for i in ids]
    tempo_medio = [por_id[i]["tempo_medio_epoca_s"] for i in ids]
    tempo_total = [por_id[i]["tempo_total_s"] / 60 for i in ids]

    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA_POL, 3.8))
    x = np.arange(len(nomes))
    largura = 0.38
    a = eixos[0, 0]
    barras_acc = a.bar(x - largura / 2, acuracia, width=largura, label="Acurácia", color="#4C78A8")
    barras_f1 = a.bar(x + largura / 2, f1, width=largura, label="F1 macro", color="#F58518")
    for barras in (barras_acc, barras_f1):
        a.bar_label(barras, labels=[virgula(v, 3) for v in barras.datavalues], fontsize=6, padding=1)
    a.set_ylim(0, 1.16)
    a.set_yticks(np.arange(0, 1.01, 0.2))
    a.set_ylabel("Valor no teste")
    a.set_title("(a) Acurácia e F1 macro no teste")
    a.legend(loc="upper center", ncol=2, fontsize=6.5, frameon=False)

    for eixo, valores, titulo, unidade, casas, cor in (
        (eixos[0, 1], memoria, "(b) Pico de memória do processo", "MB", 0, "#54A24B"),
        (eixos[1, 0], tempo_medio, "(c) Tempo médio por época", "Segundos", 1, "#4C78A8"),
        (eixos[1, 1], tempo_total, "(d) Tempo total de treinamento", "Minutos", 1, "#F58518"),
    ):
        barras = eixo.bar(nomes, valores, color=cor, width=0.6)
        rotulos = [f"{valor:,.0f}".replace(",", ".") if casas == 0 else virgula(valor, casas) for valor in valores]
        eixo.bar_label(barras, labels=rotulos, fontsize=6.5, padding=1)
        eixo.set_ylim(0, max(valores) * 1.18)
        eixo.set_ylabel(unidade)
        eixo.set_title(titulo)
    for eixo in eixos.ravel():
        eixo.set_xticks(x)
        eixo.set_xticklabels(nomes)
        eixo.grid(axis="y", alpha=0.3)
        _eixo_y_com_virgula(eixo)
    eixos[0, 1].yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:,.0f}".replace(",", ".")))
    eixos[1, 0].set_xlabel("Resolução (largura × altura)")
    eixos[1, 1].set_xlabel("Resolução (largura × altura)")
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    return _salvar(fig, destino)


def figura_erros(erros: dict, raiz: Path, destino: Path | None = None) -> Path:
    _estilo()
    destino = destino or FIGURAS / "erros_teste.png"
    amostras = erros["amostras"]
    if not amostras:
        fig, ax = plt.subplots(figsize=(LARGURA_POL, 0.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "Nenhum erro no conjunto de teste", ha="center", va="center")
        return _salvar(fig, destino)

    fig, eixos = plt.subplots(1, len(amostras), figsize=(LARGURA_POL, 2.15))
    if len(amostras) == 1:
        eixos = [eixos]
    for eixo, amostra in zip(eixos, amostras):
        with Image.open(raiz / amostra["caminho"]) as imagem:
            eixo.imshow(imagem.convert("RGB"))
        eixo.axis("off")
        eixo.set_title(
            f"{Path(amostra['caminho']).name}\n"
            f"real: {amostra['classe_real']}\n"
            f"prevista: {amostra['classe_prevista']}\n"
            f"p = {virgula(amostra['probabilidade_prevista'], 3)}",
            fontsize=6.5,
            linespacing=1.15,
        )
    fig.tight_layout(pad=0.3)
    return _salvar(fig, destino)


def gerar_todas(raiz: Path, eda: dict, divisao: dict, resultado: dict) -> None:
    figura_distribuicao(divisao["contagens"])
    figura_exemplos(raiz, eda["registros_validos"])
    figura_resolucoes(raiz, eda["registros_validos"])
    figura_curvas()
    figura_matrizes(resultado)
    figura_efeito_resolucao(resultado)
    figura_erros(resultado["erros"], raiz)
