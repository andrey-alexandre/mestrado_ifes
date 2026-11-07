# Registro de Congelamento — Amostra Definitiva do Protocolo (HAM10000, 3.000 lesões)

**Este arquivo registra, de forma permanente, a identidade exata da amostra aprovada pelo pesquisador como entrada fixa de todas as condições diagnósticas (S0/S1/S2) e de todos os sistemas/modelos avaliados no protocolo experimental.**

- Data de aprovação: 2026-11-07
- Arquivo: `C:\Projetos\IFES\mestrado_ifes\experiments\sampling\sample_manifest.csv`
- Tamanho: 126073 bytes
- **SHA-256: `32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799`**
- Semente usada na geração: `20261106`

## Por que o SHA-256, não só a semente

A semente sozinha não identifica o conteúdo do arquivo: uma mudança de código na amostragem, uma versão diferente do instantâneo de metadados do HAM10000, ou uma versão diferente do gerador de números aleatórios do Python poderiam produzir um arquivo diferente a partir da mesma semente. O SHA-256 acima é o identificador real da versão aprovada — qualquer cópia deste manifesto usada em qualquer etapa futura do protocolo deve ter exatamente este hash.

## Verificação que fundamentou a aprovação

- Relatório: `C:\Projetos\IFES\mestrado_ifes\experiments\sampling\verify_sample_manifest_report.md`
- Resultado geral: TODOS OS 4 CRITÉRIOS PASSARAM
- `lesion_id` distintos: 3000
- Divergências de cota: nenhuma
- Máscaras ausentes (S1): 0

## Decisão associada

Tarefa 11 (auditoria de duplicata visual, Tarefas 7–11) foi encerrada pelo pesquisador sem nenhuma exclusão aprovada por duplicidade — os 2 pares `indeterminado` permanecem no registro como indeterminados, não excluídos e não confirmados. Nenhuma outra exclusão foi aplicada a esta amostra.

## Regra de uso

Qualquer execução experimental (sistema multiagente, CNN baseline, qualquer condição visual) que use "a amostra do HAM10000" deve usar exatamente este arquivo, identificado pelo SHA-256 acima — não regenerar, não usar outra semente, não usar uma cópia não verificada contra este hash.
