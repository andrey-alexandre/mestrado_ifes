# 19 — Preparacao S0/S1/S2 (Etapa 5, primeira entrega)

STATUS: **implementada, aguardando execucao do pesquisador.** Nada neste
documento relata uma execucao real -- nenhum script foi corrido pelo
assistente que escreveu este codigo (`feedback_no_autonomous_execution`).
As tabelas de contagem da secao 5 sao placeholders explicitos; so devem
ser preenchidas com os numeros reais apos o pesquisador rodar
`scripts/run_s0_s1_s2_preparation.py` e devolver os dois arquivos de
saida descritos na secao 6.

## 1. Escopo desta entrega

Preparacao das tres condicoes de apresentacao de imagem usadas depois
pelos tres pipelines (VLM-direto, multiagente, CNN) da matriz
experimental (`15_multiagent_experimental_matrix.md`):

- **S0** — imagem original HAM10000, sem nenhuma alteracao alem de
  abrir/resalvar no formato de saida (sem recorte, sem redimensionamento,
  sem remocao de fundo).
- **S1** — imagem original + contorno da **mascara de referencia**
  (Tschandl/HAM10000).
- **S2** — imagem original + contorno da **mascara predita** pelo
  segmentador final (DeepLabV3+-ResNet34, checkpoint SHA-256
  `d15b712895fafac665329d138b03e9de6bd8837676bcc8f4c675042c5d206f31`,
  limiar 0,5 — ver `18_s2_final_model_closure.md`).

**Nenhuma chamada a VLM ou a qualquer modelo acontece aqui.** Esta
entrega so prepara as imagens de entrada; o stub de payload (secao 4)
e uma demonstracao de formato, nao uma chamada real.

## 2. Entradas usadas (todas ja existentes, confirmadas por leitura direta)

| Entrada | Caminho | Observacao |
|---|---|---|
| Manifesto congelado | `experiments/sampling/sample_manifest.csv` | 3.000 linhas, SHA-256 `32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`, reverificado a cada execucao (aborta se nao bater) |
| Imagens originais | `D:/Andrey/Mestrado/dataverse_files/HAM10000/{image_id}.jpg` | so as 3.000 do manifesto importam |
| Mascaras de referencia (S1) | `D:/Andrey/Mestrado/dataverse_files/HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl/{image_id}_segmentation.png` | convencao de `src/data_audit/sampling.py:308-318` |
| JSON de geracao de S2 | `experiments/segmentation/s2_mask_generation_results.json` | resultado real: 3.000/3.000, OK=3000, vazias=6, falhas=0 |
| Mascaras preditas (S2) | `experiments/segmentation/s2_masks/{image_id}_s2_mask.png` (default; caminho real a confirmar via `--s2-masks-dir`) | **ainda nao copiadas para este repositorio/maquina** -- cada arquivo e reaberto e revalidado localmente, o `status`/`mask_empty` do JSON nunca e usado sem essa reconfirmacao |

Nenhum caminho vem de uma coluna do manifesto (ele nao tem essa coluna);
todos os 3 caminhos sao sempre derivados por convencao de nome a partir
de `image_id`.

## 3. Funcao unica de contorno (S1 e S2 usam exatamente a mesma)

Implementada em `src/multiagent/presentation.py::draw_contour`. Parametros
fixos, documentados como constantes no topo do modulo (nunca redefinidos
em outro lugar com valores diferentes):

| Parametro | Valor | Justificativa |
|---|---|---|
| Cor do contorno | `(255, 0, 0)` em RGB (vermelho puro) | visualmente distinguivel sobre pele/lesao |
| Espessura | 3 px | visivel sem ocultar muito da borda da lesao |
| Modo de deteccao | `cv2.RETR_EXTERNAL` + `cv2.CHAIN_APPROX_SIMPLE` | so contorno externo de cada blob; **multiplos contornos desconexos na mesma mascara** sao tratados naturalmente (um contorno por blob), buracos internos sao ignorados |
| Anti-aliasing | NAO (`cv2.LINE_8`, nao `cv2.LINE_AA`) | borda com pixels exatos, sem blending, para nao introduzir uma diferenca visual extra entre S1 e S2 alem do proprio contorno |
| Pre-condicao | mascara deve ter exatamente o mesmo `(H, W)` da imagem original | se nao tiver, `draw_contour` levanta `ValueError` — nunca redimensiona a mascara silenciosamente |

S0 nunca passa por `draw_contour` (e so a imagem original resalva). S1
chama `draw_contour(imagem, mascara_de_referencia)`; S2 chama
`draw_contour(imagem, mascara_predita)` — mesma funcao, mesmos
parametros, unica diferenca e qual mascara entra.

## 4. Esquema de disponibilidade/motivo por condicao

Cada caso recebe, por condicao, um status (`disponivel`/`indisponivel`) e,
quando indisponivel, um motivo estrutural explicito — nunca um booleano
implicito, nunca um motivo generico "erro".

| Condicao | Coluna de status | Coluna de motivo | Motivos possiveis |
|---|---|---|---|
| S0 | `status_s0` | `reason_s0` | `imagem_original_ausente_ou_ilegivel` |
| S1 | `status_s1` | `reason_s1` | `mascara_referencia_ausente_ou_ilegivel`, `mascara_referencia_shape_incompativel`, `imagem_original_ausente_ou_ilegivel` |
| S2 | `status_s2` | `reason_s2` | `mascara_predita_vazia`, `mascara_predita_arquivo_ausente_localmente`, `mascara_predita_falha`, `mascara_predita_shape_incompativel`, `mascara_predita_sem_entrada_no_json`, `imagem_original_ausente_ou_ilegivel` |

Regras duras aplicadas pelo script:

- As 6 lesoes com `mask_empty=true` no JSON de S2 (ISIC_0030114,
  ISIC_0025985, ISIC_0024809, ISIC_0029792, ISIC_0025249, ISIC_0029514)
  tem **S0 e S1 gerados normalmente** e **S2 explicitamente
  indisponivel** (`reason_s2 = mascara_predita_vazia`) — sem fallback
  para S0/S1 nesse caso.
- Para qualquer caso com `status="ok"` e `mask_empty=false` no JSON, o
  arquivo de mascara e **reaberto e revalidado nesta maquina** antes de
  ser usado — se nao existir ou nao abrir como imagem valida, isso e
  `mascara_predita_arquivo_ausente_localmente`, uma causa DISTINTA de
  `mascara_predita_vazia` (a primeira e um problema de transporte de
  arquivo entre maquinas; a segunda e um resultado real do segmentador).
- `status="falha"` no JSON (nenhum caso real hoje, mas o codigo nao
  assume isso) -> `mascara_predita_falha`.
- Shape da mascara (referencia ou predita) diferente do shape nativo da
  imagem -> `*_shape_incompativel`; a mascara NUNCA e redimensionada
  silenciosamente para "ajustar".
- Indisponibilidade de S2 e uma categoria estrutural de **indisponibilidade
  de entrada**, nao uma recusa/indeterminacao de VLM (nao ha VLM nesta
  entrega) — um avaliador futuro deve manter esses casos fora de
  qualquer contagem de acerto/erro, nunca como "0" implicito.

Pareamento adicional registrado sem remover nenhuma linha da amostra:
`lesion_id`/`image_id` duplicados dentro do manifesto, `image_id` do
manifesto sem entrada correspondente no JSON de S2, e entradas do JSON
de S2 sem linha correspondente no manifesto (vice-versa verificado, nao
assumido vazio).

## 5. Contagens reais por condicao — PLACEHOLDER, preencher apos execucao real

**NAO preencher com numeros estimados ou "esperados" — so com o
conteudo real de `s0_s1_s2_preparation_results.json` apos a execucao do
pesquisador.**

| Condicao | Disponivel | Indisponivel | Motivos (contagem) |
|---|---|---|---|
| S0 | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| S1 | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| S2 | _(preencher)_ | _(preencher)_ | _(preencher)_ |

- Duplicatas de `lesion_id` no manifesto: _(preencher)_
- Duplicatas de `image_id` no manifesto: _(preencher)_
- `image_id` do manifesto sem entrada no JSON de S2: _(preencher)_
- Entradas do JSON de S2 sem linha correspondente no manifesto: _(preencher)_

## 6. Stub de payload dublê (sem VLM real)

`src/multiagent/presentation.py::build_vlm_payload_stub` simula o que um
agente visual receberia por caso+condicao: um dict com apenas
`image_path`, `condition` e `instruction` (placeholder genérico, sem
texto real de prompt ainda). A assinatura da funcao e deliberadamente
incapaz de aceitar `dx`/`binary_target`/identidade da mascara como
parametro — essas informacoes nao existem no escopo da funcao, entao nao
podem vazar. `verify_payload_has_no_labels` confere adicionalmente que
nenhuma chave proibida (`dx`, `binary_target`, `label`, `ground_truth`,
`mask_type`, `is_reference_mask`) ou substring (`binary_target`,
`melanoma`, `nao_melanoma`) aparece no payload.

## 7. Exemplos visuais lado a lado

O script gera, em `experiments/multiagent/s0_s1_s2_images/examples/`,
imagens compostas S0+S1+S2 (com legenda de qual condicao e qual) para
pelo menos 5 lesoes, incluindo obrigatoriamente um dos 6 casos de
mascara de S2 vazia (nesse caso o painel de S2 fica ausente da
composicao, refletindo a indisponibilidade real). Destinado a inspecao
visual do pesquisador, nao a nenhuma metrica automatica.

## 8. Comando exato para o pesquisador rodar

```
python scripts/run_s0_s1_s2_preparation.py
```

Todos os argumentos tem default apontando para os caminhos reais do
projeto; use `--s2-masks-dir` para apontar para o diretorio real onde os
3.000 PNGs de S2 estarao nessa maquina, se for diferente do default
(`experiments/segmentation/s2_masks`).

Saidas esperadas (novas, nao sobrescrevem nada existente fora dessa
pasta):
- `experiments/multiagent/s0_s1_s2_preparation_manifest.csv` (3.000
  linhas, status/motivo por condicao)
- `experiments/multiagent/s0_s1_s2_preparation_results.json` (resumo +
  contagens — usar para preencher a secao 5 deste documento)
- `experiments/multiagent/s0_s1_s2_images/{s0,s1,s2}/*.png` (imagens
  geradas)
- `experiments/multiagent/s0_s1_s2_images/examples/*.png` (compostos
  lado a lado)

## 9. Testes

```
python -m pytest tests/multiagent/test_s0_s1_s2_preparation.py -v
```

(Lembrete deste repositorio: `python -m pytest`, nao `pytest` isolado —
`python -m pytest` insere a raiz do projeto no `sys.path`.)

## 10. O que devolver para validacao

1. `experiments/multiagent/s0_s1_s2_preparation_results.json` (resumo
   real).
2. `experiments/multiagent/s0_s1_s2_preparation_manifest.csv` (ou pelo
   menos as primeiras linhas + contagem total).
3. Saida completa do comando de testes da secao 9.
4. Opcional mas recomendado: 2-3 arquivos de
   `experiments/multiagent/s0_s1_s2_images/examples/` para inspecao
   visual rapida nesta sessao.
