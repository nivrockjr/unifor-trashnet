"""Treinamento dos seis experimentos, com retomada se o modelo já existir."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENTOS_DIR,
    LEARNING_RATE,
    MIN_DELTA,
    NORMALIZACOES,
    OTIMIZADOR,
    PACIENCIA,
    PERDA,
    SEED,
    input_shape,
)
from src.dados import caminhos_e_rotulos, carregar_uint8, normalizar
from src.modelos import criar_modelo


class Lotes(keras.utils.Sequence):
    """Lotes normalizados na hora, a partir de imagens uint8 já redimensionadas."""

    def __init__(
        self,
        imagens: np.ndarray,
        rotulos: np.ndarray,
        embaralhar: bool,
        semente: int,
        lote: int = BATCH_SIZE,
        normalizacao: str = "centralizada",
    ):
        super().__init__(workers=1, use_multiprocessing=False)
        self.imagens = imagens
        self.rotulos = np.asarray(rotulos)
        self.batch_size = lote
        self.embaralhar = embaralhar
        self.normalizacao = normalizacao
        self.rng = np.random.default_rng(semente)
        self.indices = np.arange(len(imagens))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.imagens) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.embaralhar:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx: int):
        sel = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        lote = normalizar(self.imagens[sel], self.normalizacao)
        return lote, self.rotulos[sel]


class GuardarMelhor(keras.callbacks.Callback):
    """Guarda os pesos da época com menor perda de validação.

    Usa a mesma melhora mínima da parada antecipada, para a época salva ser a
    mesma que o EarlyStopping considera a melhor.
    """

    def __init__(self, melhora_minima: float):
        super().__init__()
        self.melhora_minima = melhora_minima
        self.melhor_perda = None
        self.pesos = None
        self.epoca = 0

    def on_epoch_end(self, epoch, logs=None):
        perda = float(logs["val_loss"])
        if self.melhor_perda is None or perda < self.melhor_perda - self.melhora_minima:
            self.melhor_perda = perda
            self.pesos = self.model.get_weights()
            self.epoca = int(epoch)


class CronometroEpocas(keras.callbacks.Callback):
    """Mede a duração de cada época e a CPU usada por outros processos.

    Uma thread separada compara, a cada 5 s, a CPU total da máquina com a CPU
    deste processo: a diferença é o que outros programas usaram. Um valor alto
    indica que o tempo por época foi inflado por concorrência.
    """

    INTERVALO_CPU_S = 5.0

    def __init__(self):
        super().__init__()
        self.duracoes = []
        self.cpu_outros = []
        self._inicio = None
        self._processo = psutil.Process()
        self._nucleos = psutil.cpu_count(logical=True) or 1
        self._parar = threading.Event()
        self._thread = None

    def _amostrar(self):
        while not self._parar.wait(self.INTERVALO_CPU_S):
            total = psutil.cpu_percent(interval=None)
            proprio = self._processo.cpu_percent(interval=None) / self._nucleos
            self.cpu_outros.append(max(0.0, total - proprio))

    def on_train_begin(self, logs=None):
        # Primeira leitura só zera os contadores de CPU.
        psutil.cpu_percent(interval=None)
        self._processo.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._amostrar, daemon=True)
        self._thread.start()

    def on_train_end(self, logs=None):
        self._parar.set()
        if self._thread is not None:
            self._thread.join()

    def on_epoch_begin(self, epoch, logs=None):
        self._inicio = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.duracoes.append(time.perf_counter() - self._inicio)


class _RusageInfoV4(ctypes.Structure):
    """Estrutura rusage_info_v4 do macOS (sys/resource.h), usada só para ler o pico de memória."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (nome, ctypes.c_uint64)
        for nome in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
            "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


def pico_memoria_processo() -> tuple[int | None, str]:
    """Pico de memória deste processo desde que ele começou, informado pelo sistema.

    Como scripts/treinar.py roda cada experimento num processo próprio, o pico é
    só daquele treino: inclui o TensorFlow, as imagens carregadas e o fit.
    """
    sistema = platform.system()
    try:
        if sistema == "Darwin":
            info = _RusageInfoV4()
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
                return int(info.ri_lifetime_max_phys_footprint), (
                    "macOS: ri_lifetime_max_phys_footprint, lido com proc_pid_rusage. É o pico do valor "
                    "da coluna Memória do Monitor de Atividade e conta também a memória comprimida "
                    "ou enviada ao swap, que o RSS deixa de fora."
                )
        elif sistema == "Linux":
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, (
                "Linux: ru_maxrss, lido com getrusage. Pico do RSS do processo."
            )
        elif sistema == "Windows":
            return int(psutil.Process().memory_info().peak_wset), (
                "Windows: peak_wset, lido com psutil. Pico do working set do processo."
            )
    except Exception as erro:  # a medição não pode derrubar o treino
        return None, f"Não foi possível ler o pico de memória: {erro}"
    return None, f"Sem leitura de pico de memória para o sistema {sistema}."


def nome_processador() -> str:
    """Nome comercial do processador. platform.processor() devolve só 'arm' no macOS."""
    sistema = platform.system()
    try:
        if sistema == "Darwin":
            saida = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            )
            return saida.stdout.strip()
        if sistema == "Linux":
            for linha in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if linha.lower().startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def hardware() -> dict:
    gpus = []
    try:
        for dispositivo in tf.config.list_physical_devices("GPU"):
            gpus.append(dispositivo.name)
    except Exception:
        gpus = []
    memoria = psutil.virtual_memory()
    return {
        "sistema": platform.platform(),
        "processador": nome_processador(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memoria.total / 1024**3, 2),
        "gpus": gpus,
        "observacao_gpu": (
            "Treinamento executado na CPU. O TensorFlow não encontrou GPU utilizável."
            if not gpus
            else "GPU detectada pelo TensorFlow."
        ),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }


def estado_da_maquina() -> dict:
    """RAM disponível e swap no momento."""
    return {
        "ram_disponivel_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "swap_usado_mb": round(psutil.swap_memory().used / 1024**2, 1),
    }


def preparar_tensorflow() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except RuntimeError:
        pass


def pesos_de_classe(rotulos: np.ndarray) -> dict[int, float]:
    """Pesos inversamente proporcionais à frequência, só com o conjunto de treino."""
    classes = np.arange(6)
    valores = compute_class_weight(class_weight="balanced", classes=classes, y=rotulos)
    return {int(classe): float(peso) for classe, peso in zip(classes, valores)}


def agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_treinos_neste_processo = 0


def treinar_um(
    espec: dict,
    raiz_dataset: Path,
    divisao: dict,
    substituir: bool = False,
    pasta_base: Path = EXPERIMENTOS_DIR,
) -> dict:
    global _treinos_neste_processo
    preparar_tensorflow()
    pasta = pasta_base / espec["id"]
    caminho_modelo = pasta / "modelo.keras"
    caminho_registro = pasta / "registro.json"
    if caminho_modelo.is_file() and caminho_registro.is_file() and not substituir:
        print(f"[pular] {espec['id']} já possui modelo salvo.", flush=True)
        return json.loads(caminho_registro.read_text(encoding="utf-8"))
    pasta.mkdir(parents=True, exist_ok=True)
    _treinos_neste_processo += 1

    normalizacao = espec.get("normalizacao", "centralizada")
    forma = input_shape(espec["resolucao"])
    altura, largura, _ = forma
    treino_caminhos, treino_y = caminhos_e_rotulos(raiz_dataset, divisao["partes"]["treino"])
    val_caminhos, val_y = caminhos_e_rotulos(raiz_dataset, divisao["partes"]["validacao"])

    print(f"[dados] {agora()} carregando {espec['id']} em {largura}×{altura}...", flush=True)
    inicio_carga = time.perf_counter()
    treino_x = carregar_uint8(treino_caminhos, altura, largura)
    val_x = carregar_uint8(val_caminhos, altura, largura)
    duracao_carga = time.perf_counter() - inicio_carga
    lotes_treino = Lotes(treino_x, treino_y, True, SEED, BATCH_SIZE, normalizacao)
    lotes_val = Lotes(val_x, val_y, False, SEED, BATCH_SIZE, normalizacao)

    tf.keras.utils.set_random_seed(SEED)
    modelo = criar_modelo(espec["arquitetura"], forma)
    pesos = pesos_de_classe(treino_y) if espec["pesos_classe"] else None

    cronometro = CronometroEpocas()
    guardar = GuardarMelhor(MIN_DELTA)
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PACIENCIA,
        min_delta=MIN_DELTA,
        restore_best_weights=False,
        verbose=1,
    )
    maquina_inicio = estado_da_maquina()
    inicio_iso = agora()
    print(f"[treino] {inicio_iso} início de {espec['id']}, lote {BATCH_SIZE}, até {EPOCHS} épocas", flush=True)
    inicio = time.perf_counter()
    historico = modelo.fit(
        lotes_treino,
        validation_data=lotes_val,
        epochs=EPOCHS,
        shuffle=False,
        class_weight=pesos,
        callbacks=[parada, cronometro, guardar],
        verbose=2,
    )
    duracao_fit = time.perf_counter() - inicio
    if _treinos_neste_processo == 1:
        pico_memoria, medicao_memoria = pico_memoria_processo()
    else:
        pico_memoria, medicao_memoria = None, (
            "Não medido: houve mais de um treino neste processo, e o pico de um se misturaria com o do outro."
        )
    fim_iso = agora()
    maquina_fim = estado_da_maquina()
    parametros = int(modelo.count_params())
    modelo.set_weights(guardar.pesos)
    modelo.save(caminho_modelo)
    del treino_x, val_x, lotes_treino, lotes_val, modelo
    try:
        keras.backend.clear_session()
    except Exception:
        pass

    historico_serial = {chave: [float(v) for v in valores] for chave, valores in historico.history.items()}
    epocas_rodadas = len(historico_serial.get("loss", []))
    melhor_epoca_base0 = int(guardar.epoca)
    parou_antes_do_teto = epocas_rodadas < EPOCHS
    registro = {
        "id": espec["id"],
        "arquitetura": espec["arquitetura"],
        "resolucao": espec["resolucao"],
        "input_shape_hwc": list(forma),
        "pesos_classe": espec["pesos_classe"],
        "pesos": pesos,
        "semente": SEED,
        "otimizador": OTIMIZADOR,
        "taxa_aprendizagem": LEARNING_RATE,
        "perda": PERDA,
        "lote": BATCH_SIZE,
        "epocas_maximas": EPOCHS,
        "paciencia": PACIENCIA,
        "melhora_minima": MIN_DELTA,
        "monitor_parada": "val_loss",
        "criterio_parada": (
            f"Parada antecipada na perda de validação (paciência {PACIENCIA}, melhora mínima {MIN_DELTA}). "
            "O arquivo salvo corresponde à época de menor perda de validação. "
            "A seleção entre os seis modelos usa o F1 macro na validação, calculado depois do treino."
        ),
        "epocas_executadas": epocas_rodadas,
        "parou_pela_paciencia": parou_antes_do_teto,
        "epoca_modelo_salvo": melhor_epoca_base0 + 1,
        "epoca_modelo_salvo_base0": melhor_epoca_base0,
        "inicio": inicio_iso,
        "fim": fim_iso,
        "tempo_carga_imagens_s": float(duracao_carga),
        "tempo_total_s": float(sum(cronometro.duracoes)),
        "tempo_fit_s": float(duracao_fit),
        "tempo_medio_epoca_s": float(np.mean(cronometro.duracoes)) if cronometro.duracoes else None,
        "tempos_epoca_s": [float(v) for v in cronometro.duracoes],
        "parametros": parametros,
        "tamanho_modelo_bytes": int(caminho_modelo.stat().st_size),
        "pico_memoria_bytes": pico_memoria,
        "maquina_inicio": maquina_inicio,
        "maquina_fim": maquina_fim,
        "cpu_outros_processos_pct": {
            "media": round(float(np.mean(cronometro.cpu_outros)), 1) if cronometro.cpu_outros else None,
            "maximo": round(float(np.max(cronometro.cpu_outros)), 1) if cronometro.cpu_outros else None,
            "amostras": len(cronometro.cpu_outros),
            "metodo": (
                "A cada 5 s: CPU total da máquina menos a CPU deste processo, em % da capacidade total. "
                "Mede o quanto outros programas disputaram a CPU durante o treino."
            ),
        },
        "hardware": hardware(),
        "medicao_tempo": (
            "Cada época é medida com time.perf_counter entre on_epoch_begin e on_epoch_end, "
            "incluindo o passo de validação. O tempo total é a soma dessas durações. "
            "Ficam de fora o download, a divisão e a carga e o redimensionamento das imagens, "
            "registrados à parte em tempo_carga_imagens_s."
        ),
        "medicao_memoria": medicao_memoria,
        "normalizacao": NORMALIZACOES[normalizacao],
        "historico": historico_serial,
    }
    caminho_registro.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    pico_texto = f"{pico_memoria / 1024**2:.0f} MB" if pico_memoria else "não medido"
    print(
        f"[ok] {fim_iso} {espec['id']}: {epocas_rodadas} épocas, "
        f"modelo da época {registro['epoca_modelo_salvo']}, "
        f"{registro['tempo_total_s']:.1f}s, pico de memória {pico_texto}",
        flush=True,
    )
    return registro
