# primeiro_trabalho_po

1º Trabalho de Pesquisa Operacional (UFF) — quadro de horários modelado como Programa Inteiro Misto (PuLP + HiGHS).

## Instalação

```
pip install -r requirements.txt
```

## Execução

```
python horarios.py                 # cenário base: 4 salas, 1 dia/professor, fator 2
python horarios.py --salas 6       # outro número de salas
python horarios.py --dias 2        # professores podem dar aula em até 2 dias
python horarios.py --fator 2.5     # fator de equidade entre penalidades
python horarios.py --tempo 3600 --log
python cenarios.py                 # roda todas as análises e gera resultados/ (tabelas e gráficos)
```

O modelo é gravado em `modelo.lp`.
