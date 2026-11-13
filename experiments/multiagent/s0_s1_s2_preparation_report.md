# Preparacao S0/S1/S2 — HAM10000 (sem chamada a VLM)

- Manifesto: `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/sampling/sample_manifest.csv`
- Total de casos: **3000**
- Relatorio por caso (CSV): `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/multiagent/s0_s1_s2_preparation_manifest.csv`

## Contagens por condicao

### S0

- Disponivel: 3000
- Indisponivel: 0
- Soma (deve ser 3000): 3000

### S1

- Disponivel: 3000
- Indisponivel: 0
- Soma (deve ser 3000): 3000

### S2

- Disponivel: 2994
- Indisponivel: 6
- Soma (deve ser 3000): 3000

| Motivo | Contagem |
|---|---|
| `mascara_predita_vazia` | 6 |

## Duplicatas e divergencias manifesto <-> JSON de S2 (registro, sem exclusao)

- `lesion_id` duplicados: nenhum
- `image_id` duplicados: nenhum
- `image_id` do manifesto sem entrada no JSON de S2: nenhum
- Entradas do JSON de S2 sem linha correspondente no manifesto: nenhuma

## Aviso

Este relatorio e gerado por `scripts/run_s0_s1_s2_preparation.py` sobre o manifesto congelado; nenhuma chamada a VLM/LLM ocorre nesta entrega. Nenhum caso e descartado ou substituido em nenhuma das 3 condicoes.
