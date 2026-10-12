"""
Análise de cenários do 1º Trabalho de Pesquisa Operacional.

Cada análise altera exclusivamente um parâmetro em relação ao modelo base
(4 salas, 1 dia por professor, fator 2). Os resultados são gravados em
resultados/cenarios.json (o script retoma de onde parou) e são gerados
tabelas e gráficos em resultados/.

Uso:
    python cenarios.py              # executa os cenários que faltam e gera os gráficos
    python cenarios.py --so-graficos
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from horarios import resolver

PASTA = "resultados"
ARQ = os.path.join(PASTA, "cenarios.json")
BASE = {"salas": 4, "max_dias": 1, "fator": 2.0}

ANALISES = {
    "salas": ("Número de salas", [4, 5, 6, 7, 8]),
    "max_dias": ("Máximo de dias por professor", [1, 2, 3, 4]),
    "fator": ("Fator máximo entre penalidades", [2.0, 2.25, 2.5, 2.75, 3.0]),
}


def chave(c):
    return f"salas={c['salas']}|dias={c['max_dias']}|fator={c['fator']}"


def carregar():
    if os.path.exists(ARQ):
        with open(ARQ, encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar(res):
    with open(ARQ, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def executar(dados, tempo):
    res = carregar()
    for param, (_, valores) in ANALISES.items():
        for v in valores:
            cen = dict(BASE, **{param: v})
            k = chave(cen)
            if k in res:
                continue
            print(f"Resolvendo {k} ...", flush=True)
            r = resolver(dados, cen["salas"], cen["max_dias"], cen["fator"], tempo)
            r.pop("quadro", None)
            res[k] = r
            salvar(res)
            print(f"  status={r['status_sol']}  FO={r['fo']}  tempo={r['tempo']:.1f}s", flush=True)
            if r.get("violacoes"):
                print("  ATENÇÃO, violações: " + "; ".join(r["violacoes"]), flush=True)
    return res


def graficos(res):
    linhas_md = []
    for param, (titulo, valores) in ANALISES.items():
        regs = [res.get(chave(dict(BASE, **{param: v}))) for v in valores]
        pares = [(v, r) for v, r in zip(valores, regs) if r]
        if not pares:
            continue
        xs = [str(v) for v, _ in pares]
        # cenário sem solução viável aparece como barra vazia
        fos = [r["fo"] if r.get("penalidades") else None for _, r in pares]
        tempos = [r["tempo"] for _, r in pares]

        linhas_md.append(f"\n### {titulo}\n")
        linhas_md.append(f"| {titulo} | Status | Função-objetivo | Gap | Tempo (s) | Pen. mín | Pen. máx |")
        linhas_md.append("|---|---|---|---|---|---|---|")
        for (v, r), fo in zip(pares, fos):
            pens = r.get("penalidades")
            gap = f"{100 * r['gap']:.2f}%" if r.get("gap") is not None and fo is not None else "-"
            linhas_md.append(f"| {v} | {r['status_sol']} | {f'{fo:.0f}' if fo is not None else '-'} | "
                             f"{gap} | {r['tempo']:.1f} | "
                             f"{min(pens) if pens else '-'} | {max(pens) if pens else '-'} |")

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.bar(xs, [f if f is not None else 0 for f in fos], color="#4C72B0", label="Função-objetivo")
        ax1.set_xlabel(titulo)
        ax1.set_ylabel("Função-objetivo (penalidade total)")
        for i, f in enumerate(fos):
            if f is not None:
                ax1.annotate(f"{f:.0f}", (i, f), ha="center", va="bottom")
        ax2 = ax1.twinx()
        ax2.plot(xs, tempos, color="#DD8452", marker="o", label="Tempo (s)")
        ax2.set_ylabel("Tempo computacional (s)")
        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
        ax1.set_title(f"Impacto: {titulo.lower()}")
        fig.tight_layout()
        fig.savefig(os.path.join(PASTA, f"cenario_{param}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(PASTA, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md) + "\n")
    print("\n".join(linhas_md))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="dados_trab_modelagem.txt")
    ap.add_argument("--tempo", type=float, default=3600)
    ap.add_argument("--so-graficos", action="store_true")
    args = ap.parse_args()
    os.makedirs(PASTA, exist_ok=True)
    res = carregar() if args.so_graficos else executar(args.dados, args.tempo)
    graficos(res)


if __name__ == "__main__":
    main()
