# Piloto local de VLM — medgemma1.5:4b-it-q4_K_M — ESTÁGIO 1 PARADO (stage=`controle_de_envio_de_imagem`)

- Etapa em que parou: `controle_de_envio_de_imagem`
- Matriz principal de 92 chamadas: NÃO tentada.
- Log de chamadas (controles): `/Users/andrey.basilio/Projetos/master/mestrado_ifes/experiments/multiagent/local_vlm_pilot/medgemma1.5_4b-it-q4_K_M/call_log.jsonl`
- Total de chamadas planejadas por modelo (controles + matriz, teto operacional de 4h): 108
- Parser OFICIAL de classificação/gate (decisão 2026-11-14): `v2_conclusion_line_positivo_suspeita_melanoma_qualificadores_2026_11_14` (v1/`classify_case_status` mantido só para rastreabilidade -- nunca fallback/desempate)

## Controles

```json
{
  "image_sending": {
    "per_role": {
      "vlm_direto": {
        "per_control": [
          {
            "name": "controle_imagem_vlm_direto_1_vermelho_superior-esquerdo",
            "role": "vlm_direto",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": true,
            "passed": true,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_vlm_direto_2_azul_inferior-direito",
            "role": "vlm_direto",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": true,
            "passed": true,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_vlm_direto_3_sem_imagem",
            "role": "vlm_direto",
            "kind": "sem_imagem",
            "acknowledged_no_image": true,
            "passed": true,
            "payload_correct": true
          }
        ],
        "all_individually_passed": true,
        "all_payloads_correct": true,
        "responses_vary_with_image_content": true,
        "overall_pass": true
      },
      "diagnostic_abcd": {
        "per_control": [
          {
            "name": "controle_imagem_diagnostic_abcd_1_vermelho_superior-esquerdo",
            "role": "diagnostic_abcd",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": true,
            "passed": true,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_abcd_2_azul_inferior-direito",
            "role": "diagnostic_abcd",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": false,
            "passed": false,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_abcd_3_sem_imagem",
            "role": "diagnostic_abcd",
            "kind": "sem_imagem",
            "acknowledged_no_image": true,
            "passed": true,
            "payload_correct": true
          }
        ],
        "all_individually_passed": false,
        "all_payloads_correct": true,
        "responses_vary_with_image_content": true,
        "overall_pass": false
      },
      "diagnostic_menzies": {
        "per_control": [
          {
            "name": "controle_imagem_diagnostic_menzies_1_vermelho_superior-esquerdo",
            "role": "diagnostic_menzies",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": true,
            "passed": true,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_menzies_2_azul_inferior-direito",
            "role": "diagnostic_menzies",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": false,
            "passed": false,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_menzies_3_sem_imagem",
            "role": "diagnostic_menzies",
            "kind": "sem_imagem",
            "acknowledged_no_image": true,
            "passed": true,
            "payload_correct": true
          }
        ],
        "all_individually_passed": false,
        "all_payloads_correct": true,
        "responses_vary_with_image_content": true,
        "overall_pass": false
      },
      "diagnostic_spcl": {
        "per_control": [
          {
            "name": "controle_imagem_diagnostic_spcl_1_vermelho_superior-esquerdo",
            "role": "diagnostic_spcl",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": true,
            "passed": true,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_spcl_2_azul_inferior-direito",
            "role": "diagnostic_spcl",
            "kind": "imagem",
            "color_found": true,
            "quadrant_found": false,
            "passed": false,
            "payload_correct": true
          },
          {
            "name": "controle_imagem_diagnostic_spcl_3_sem_imagem",
            "role": "diagnostic_spcl",
            "kind": "sem_imagem",
            "acknowledged_no_image": true,
            "passed": true,
            "payload_correct": true
          }
        ],
        "all_individually_passed": false,
        "all_payloads_correct": true,
        "responses_vary_with_image_content": true,
        "overall_pass": false
      }
    },
    "n_roles_evaluated": 4,
    "n_roles_passed": 1,
    "overall_pass": false
  }
}
```
