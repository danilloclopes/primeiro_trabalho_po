"""
1º Trabalho de Pesquisa Operacional - UFF
Montagem do quadro de horários como Programa Inteiro Misto (PIM).

Uso:
    python horarios.py                      # cenário base (4 salas, 1 dia, fator 2)
    python horarios.py --salas 6            # altera o número de salas
    python horarios.py --dias 3             # altera o nº máximo de dias por professor
    python horarios.py --fator 2.5          # altera o fator de equidade das penalidades
    python horarios.py --tempo 3600 --lp modelo.lp
"""

import argparse
import sys
import time

import pulp

NUM_DIAS = 4
NUM_HORARIOS = 6
NUM_SLOTS = NUM_DIAS * NUM_HORARIOS
AULAS_POR_DISCIPLINA = 2
NOMES_DIAS = ["Dia 1", "Dia 2", "Dia 3", "Dia 4"]


def ler_dados(caminho):
    """Lê o arquivo texto e devolve (disciplinas por professor, disciplinas por período,
    penalidades por professor)."""
    blocos = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith("#"):
                blocos.append([])
            else:
                blocos[-1].append([int(v) for v in linha.split()])
    prof_disc, periodo_disc, penalidades = blocos
    return prof_disc, periodo_disc, penalidades


def construir_modelo(prof_disc, periodo_disc, penal, salas=4, max_dias=1, fator=2.0):
    P = range(len(prof_disc))
    T = range(NUM_SLOTS)
    K = range(NUM_DIAS)
    disciplinas = sorted(d for ds in prof_disc for d in ds)
    dia = lambda t: t // NUM_HORARIOS

    m = pulp.LpProblem("QuadroDeHorarios", pulp.LpMinimize)

    # --- Variáveis ---
    # x[d,t] = 1 se a disciplina d tem aula no horário t (binária)
    x = pulp.LpVariable.dicts("x", (disciplinas, T), cat="Binary")
    # y[p,k] = 1 se o professor p dá aula no dia k (binária)
    y = pulp.LpVariable.dicts("y", (P, K), cat="Binary")
    # pen[p] = penalidade total do professor p (contínua)
    pen = pulp.LpVariable.dicts("pen", P, lowBound=0, cat="Continuous")
    # maior e menor penalidade entre os professores (contínuas)
    pmax = pulp.LpVariable("pen_max", lowBound=0, cat="Continuous")
    pmin = pulp.LpVariable("pen_min", lowBound=0, cat="Continuous")

    # --- Função-objetivo: soma das penalidades de todos os professores ---
    m += pulp.lpSum(pen[p] for p in P), "PenalidadeTotal"

    # Definição da penalidade de cada professor
    for p in P:
        m += pen[p] == pulp.lpSum(penal[p][t] * x[d][t] for d in prof_disc[p] for t in T), f"DefPen_{p}"

    # (2) Cada disciplina possui 2 aulas semanais
    for d in disciplinas:
        m += pulp.lpSum(x[d][t] for t in T) == AULAS_POR_DISCIPLINA, f"Aulas_{d}"

    # (1) Cada professor dá no máximo uma aula por horário
    for p in P:
        for t in T:
            m += pulp.lpSum(x[d][t] for d in prof_disc[p]) <= 1, f"Prof_{p}_{t}"

    # (3) Disciplinas do mesmo período não podem ocorrer no mesmo horário
    for i, ds in enumerate(periodo_disc):
        for t in T:
            m += pulp.lpSum(x[d][t] for d in ds) <= 1, f"Periodo_{i}_{t}"

    # (4) Número de salas disponíveis em cada horário
    for t in T:
        m += pulp.lpSum(x[d][t] for d in disciplinas) <= salas, f"Salas_{t}"

    # (5) Cada professor dá aula em no máximo `max_dias` dias da semana
    for p in P:
        for d in prof_disc[p]:
            for t in T:
                m += x[d][t] <= y[p][dia(t)], f"Dia_{p}_{d}_{t}"
        m += pulp.lpSum(y[p][k] for k in K) <= max_dias, f"MaxDias_{p}"

    # Desigualdades válidas (não mudam o ótimo, apenas fortalecem a relaxação linear):
    # o professor dá 4 aulas na semana, logo no máximo 4 em cada dia que trabalha
    for p in P:
        aulas_prof = AULAS_POR_DISCIPLINA * len(prof_disc[p])
        for k in K:
            m += pulp.lpSum(x[d][t] for d in prof_disc[p] for t in T if dia(t) == k) \
                 <= aulas_prof * y[p][k], f"AulasDia_{p}_{k}"
    # com um único dia, todas as 4 aulas caem nele: a penalidade é no mínimo
    # a soma das 4 menores penalidades do professor naquele dia
    if max_dias == 1:
        for p in P:
            aulas_prof = AULAS_POR_DISCIPLINA * len(prof_disc[p])
            menor = [sum(sorted(penal[p][k * NUM_HORARIOS:(k + 1) * NUM_HORARIOS])[:aulas_prof])
                     for k in K]
            m += pen[p] >= pulp.lpSum(menor[k] * y[p][k] for k in K), f"PenMinDia_{p}"

    # (6) Penalidade de um professor <= fator * penalidade de qualquer outro.
    # Equivale a max_p pen[p] <= fator * min_p pen[p].
    for p in P:
        m += pen[p] <= pmax, f"PenMax_{p}"
        m += pen[p] >= pmin, f"PenMin_{p}"
    m += pmax <= fator * pmin, "Equidade"

    return m, x, pen


def resolver(caminho_dados, salas=4, max_dias=1, fator=2.0, tempo=3600, lp=None, msg=False):
    prof_disc, periodo_disc, penal = ler_dados(caminho_dados)
    m, x, pen = construir_modelo(prof_disc, periodo_disc, penal, salas, max_dias, fator)
    if lp:
        m.writeLP(lp)

    solver = pulp.HiGHS(msg=msg, timeLimit=tempo)
    inicio = time.perf_counter()
    m.solve(solver)
    duracao = time.perf_counter() - inicio

    resultado = {
        "salas": salas,
        "max_dias": max_dias,
        "fator": fator,
        "status": pulp.LpStatus[m.status],
        "status_sol": pulp.constants.LpSolution[m.sol_status],
        "fo": pulp.value(m.objective),
        "tempo": duracao,
        "limitante": None,
        "gap": None,
    }
    try:  # limitante inferior e gap informados pelo HiGHS (relevante se o tempo limite estourar)
        info = m.solverModel.getInfo()
        resultado["limitante"] = info.mip_dual_bound
        resultado["gap"] = info.mip_gap
    except AttributeError:
        pass
    if m.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        resultado["quadro"] = extrair_quadro(prof_disc, periodo_disc, x)
        resultado["penalidades"] = [round(pen[p].value()) for p in range(len(prof_disc))]
        resultado["violacoes"] = verificar_solucao(resultado["quadro"], prof_disc, periodo_disc,
                                                   penal, salas, max_dias, fator)
    return resultado


def verificar_solucao(quadro, prof_disc, periodo_disc, penal, salas, max_dias, fator):
    """Confere, de forma independente do modelo, se o quadro respeita as 6 regras.
    Devolve a lista de violações encontradas (vazia se o quadro é válido)."""
    erros = []
    aulas = {d: [] for ds in prof_disc for d in ds}
    for t, lista in quadro.items():
        for d, _, _ in lista:
            aulas[d].append(t)
    for p, ds in enumerate(prof_disc):
        horarios = [t for d in ds for t in aulas[d]]
        if len(horarios) != len(set(horarios)):
            erros.append(f"Regra 1: professor {p} com duas aulas no mesmo horário")
        if len({t // NUM_HORARIOS for t in horarios}) > max_dias:
            erros.append(f"Regra 5: professor {p} dá aula em mais de {max_dias} dia(s)")
    for d, ts in aulas.items():
        if len(ts) != AULAS_POR_DISCIPLINA:
            erros.append(f"Regra 2: disciplina {d} com {len(ts)} aulas")
    for t, lista in quadro.items():
        periodos = [per for _, _, per in lista]
        if len(periodos) != len(set(periodos)):
            erros.append(f"Regra 3: período repetido no horário {t}")
        if len(lista) > salas:
            erros.append(f"Regra 4: {len(lista)} aulas no horário {t} com {salas} salas")
    pens = [sum(penal[p][t] for d in ds for t in aulas[d]) for p, ds in enumerate(prof_disc)]
    if max(pens) > fator * min(pens) + 1e-6:
        erros.append(f"Regra 6: penalidade máxima {max(pens)} > {fator} x mínima {min(pens)}")
    return erros


def extrair_quadro(prof_disc, periodo_disc, x):
    prof_de = {d: p for p, ds in enumerate(prof_disc) for d in ds}
    periodo_de = {d: i for i, ds in enumerate(periodo_disc) for d in ds}
    quadro = {t: [] for t in range(NUM_SLOTS)}
    for d in prof_de:
        for t in range(NUM_SLOTS):
            if x[d][t].value() > 0.5:
                quadro[t].append((d, prof_de[d], periodo_de[d]))
    # Qualquer sala serve para qualquer disciplina: as salas são numeradas na ordem
    for t in quadro:
        quadro[t].sort(key=lambda a: a[2])
    return quadro


def imprimir_quadro(res):
    quadro = res["quadro"]
    print("\n=== Quadro de horários (por período) ===")
    largura = 16
    for i in range(6):
        print(f"\n{i + 1}º período")
        print(" " * 9 + "".join(f"{n:<{largura}}" for n in NOMES_DIAS))
        for h in range(NUM_HORARIOS):
            linha = f"Hor. {h + 1}  "
            for k in range(NUM_DIAS):
                t = k * NUM_HORARIOS + h
                cel = next((f"D{d:02d} (P{p:02d})" for d, p, per in quadro[t] if per == i), "-")
                linha += f"{cel:<{largura}}"
            print(linha)

    print("\n=== Quadro de horários (por sala) ===")
    print(" " * 9 + "".join(f"{n:<{largura}}" for n in NOMES_DIAS))
    for h in range(NUM_HORARIOS):
        for s in range(res["salas"]):
            linha = f"H{h + 1} S{s + 1}  "
            for k in range(NUM_DIAS):
                aulas = quadro[k * NUM_HORARIOS + h]
                cel = f"D{aulas[s][0]:02d} (P{aulas[s][1]:02d})" if s < len(aulas) else "-"
                linha += f"{cel:<{largura}}"
            print(linha)

    pens = res["penalidades"]
    print("\nPenalidade por professor:")
    print("  " + "  ".join(f"P{p:02d}={v}" for p, v in enumerate(pens)))
    razao = f"{max(pens) / min(pens):.3f}" if min(pens) > 0 else "-"
    print(f"  mínima={min(pens)}  máxima={max(pens)}  razão={razao}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap =argparse.ArgumentParser(description="Quadro de horários via PIM")
    ap.add_argument("--dados", default="dados_trab_modelagem.txt")
    ap.add_argument("--salas", type=int, default=4)
    ap.add_argument("--dias", type=int, default=1, help="nº máximo de dias por professor")
    ap.add_argument("--fator", type=float, default=2.0, help="fator de equidade das penalidades")
    ap.add_argument("--tempo", type=float, default=3600, help="tempo limite do solver (s)")
    ap.add_argument("--lp", default="modelo.lp", help="arquivo .lp gerado")
    ap.add_argument("--log", action="store_true", help="mostra o log do solver")
    args = ap.parse_args()

    res = resolver(args.dados, args.salas, args.dias, args.fator, args.tempo, args.lp, args.log)
    print(f"Cenário: salas={res['salas']}  dias/professor={res['max_dias']}  fator={res['fator']}")
    if res["status_sol"] == "Optimal Solution Found":
        situacao = "solução ótima (comprovada)"
    elif "quadro" in res:
        situacao = "tempo limite atingido: melhor solução encontrada, sem prova de otimalidade"
    else:
        situacao = res["status"]
    print(f"Status: {situacao}")
    if "quadro" in res:
        imprimir_quadro(res)
        if res["violacoes"]:
            print("\nATENÇÃO, o quadro viola regras:\n  " + "\n  ".join(res["violacoes"]))
        else:
            print("\nVerificação: o quadro respeita as regras 1 a 6.")
        print(f"\nValor da função-objetivo: {res['fo']:.0f}")
        if res["status_sol"] != "Optimal Solution Found" and res["limitante"] is not None:
            print(f"Limitante inferior: {res['limitante']:.2f}  (gap {100 * res['gap']:.2f}%)")
    else:
        print("Nenhuma solução viável foi encontrada.")
    print(f"Tempo de processamento: {res['tempo']:.2f} s")


if __name__ == "__main__":
    main()
