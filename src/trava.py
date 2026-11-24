"""Trava contra dois treinos ao mesmo tempo na mesma pasta."""

from __future__ import annotations

import os
from contextlib import contextmanager

import psutil

from src.config import TRAVA_TREINO


def _dono_da_trava() -> int:
    try:
        return int(TRAVA_TREINO.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


@contextmanager
def trava_de_treino():
    """Impede dois treinos ao mesmo tempo na mesma pasta.

    Dois processos dividindo a CPU distorcem o tempo por época. Se a trava existir
    e o processo dono ainda estiver vivo, o novo treino não começa. A exceção é um
    processo filho do dono: é assim que scripts/treinar.py roda cada experimento
    num processo próprio enquanto segura a trava.
    """
    TRAVA_TREINO.parent.mkdir(parents=True, exist_ok=True)
    if TRAVA_TREINO.exists():
        dono = _dono_da_trava()
        if dono and dono == os.getppid():
            yield
            return
        if dono and psutil.pid_exists(dono):
            raise SystemExit(
                f"Já existe um treino em andamento (processo {dono}). "
                "Espere ele terminar; dois treinos juntos distorcem a medição de tempo."
            )
        TRAVA_TREINO.unlink(missing_ok=True)
    try:
        arquivo = TRAVA_TREINO.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit("Outro treino acabou de começar. Espere ele terminar.") from None
    with arquivo:
        arquivo.write(str(os.getpid()))
    try:
        yield
    finally:
        TRAVA_TREINO.unlink(missing_ok=True)
