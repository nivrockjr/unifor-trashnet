"""Treina os seis experimentos controlados, cada um num processo Python próprio.

Sem --id, este script segura a trava de treino e chama a si mesmo com --id para
cada experimento, um de cada vez. Com um processo por experimento, o pico de
memória informado pelo sistema é só daquele treino. Pula modelos já salvos.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXPERIMENTOS, EXPERIMENTOS_DIR
from src.trava import trava_de_treino


def ja_treinado(ident: str) -> bool:
    pasta = EXPERIMENTOS_DIR / ident
    return (pasta / "modelo.keras").is_file() and (pasta / "registro.json").is_file()


def treinar_neste_processo(ident: str, substituir: bool) -> None:
    # Importações pesadas (TensorFlow) só no processo que treina.
    from src.config import ARTIFACTS
    from src.dados import carregar_json, extrair_zip_se_necessario
    from src.treino import treinar_um

    divisao = carregar_json(ARTIFACTS / "divisao.json")
    raiz = extrair_zip_se_necessario()
    espec = next(item for item in EXPERIMENTOS if item["id"] == ident)
    treinar_um(espec, raiz, divisao, substituir=substituir)


def treinar_em_processo_proprio(ident: str, substituir: bool) -> int:
    comando = [sys.executable, "-u", str(Path(__file__).resolve()), "--id", ident]
    if substituir:
        comando.append("--substituir")
    return subprocess.run(comando, check=False).returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treina os experimentos do TrashNet.")
    parser.add_argument("--id", choices=[item["id"] for item in EXPERIMENTOS], help="Treina só um experimento.")
    parser.add_argument("--substituir", action="store_true", help="Refaz o treino mesmo se o modelo existir.")
    args = parser.parse_args(argv)

    with trava_de_treino():
        if args.id:
            treinar_neste_processo(args.id, args.substituir)
            return
        for espec in EXPERIMENTOS:
            if ja_treinado(espec["id"]) and not args.substituir:
                print(f"[pular] {espec['id']} já possui modelo salvo.", flush=True)
                continue
            saida = treinar_em_processo_proprio(espec["id"], args.substituir)
            if saida != 0:
                raise SystemExit(f"O treino de {espec['id']} falhou (saída {saida}).")


if __name__ == "__main__":
    main()
