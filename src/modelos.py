"""Arquiteturas MLP e CNN usadas nos experimentos controlados."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import LEARNING_RATE, PERDA, SEED


def _compilar(modelo: keras.Model) -> keras.Model:
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=PERDA,
        metrics=["accuracy"],
    )
    return modelo


def criar_mlp(input_shape: tuple[int, int, int]) -> keras.Model:
    """MLP sobre pixels RGB vetorizados, com duas camadas ocultas e ReLU.

    O vetor de entrada em 128×96 tem 36.864 atributos e o treino tem pouco
    mais de 1,7 mil imagens. Dropout de 0,3 reduz a coadaptação dos neurônios.
    A entrada já chega centralizada em torno de zero.
    """
    tf.keras.utils.set_random_seed(SEED)
    entradas = keras.Input(shape=input_shape, name="imagem")
    x = layers.Flatten(name="vetor")(entradas)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", name="oculta_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="oculta_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    saidas = layers.Dense(6, activation="softmax", name="classes")(x)
    modelo = keras.Model(entradas, saidas, name="mlp_trashnet")
    return _compilar(modelo)


def criar_cnn(input_shape: tuple[int, int, int]) -> keras.Model:
    """CNN com três blocos convolucionais e global average pooling.

    O pooling global fixa o tamanho do vetor que chega ao classificador.
    Assim, a quantidade de parâmetros permanece a mesma em 128×96, 256×192
    e 512×384, e a comparação de custo reflete sobretudo a resolução.
    """
    tf.keras.utils.set_random_seed(SEED)

    def bloco(x, filtros: int, nome: str):
        x = layers.Conv2D(
            filtros,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{nome}_conv",
        )(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"{nome}_pool")(x)
        return x

    entradas = keras.Input(shape=input_shape, name="imagem")
    x = bloco(entradas, 32, "bloco1")
    x = bloco(x, 64, "bloco2")
    x = bloco(x, 128, "bloco3")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="densa")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    saidas = layers.Dense(6, activation="softmax", name="classes")(x)
    modelo = keras.Model(entradas, saidas, name="cnn_trashnet")
    return _compilar(modelo)


def criar_modelo(arquitetura: str, input_shape: tuple[int, int, int]) -> keras.Model:
    if arquitetura == "mlp":
        return criar_mlp(input_shape)
    if arquitetura == "cnn":
        return criar_cnn(input_shape)
    raise ValueError(f"Arquitetura desconhecida: {arquitetura}")
