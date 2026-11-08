# Piloto: remoção do RAG incondicional dos entry points

Verificação local com respostas simuladas, sem rede/credenciais. Confirma que run_pipeline.py e run_train.py não constroem RAG no caminho oficial (sem --build-rag) e que o caminho histórico (--build-rag) continua funcionando.

- `run_pipeline` / caminho_oficial: {'entrypoint': 'run_pipeline', 'check': 'caminho_oficial', 'retriever_is_none': True, 'ok': True, 'error': None}
- `run_pipeline` / caminho_historico: {'entrypoint': 'run_pipeline', 'check': 'caminho_historico', 'ok': True, 'error': None, 'retriever_is_sentinel': True}
- `run_train` / caminho_oficial: {'entrypoint': 'run_train', 'check': 'caminho_oficial', 'retriever_is_none': True, 'ok': True, 'error': None}
- `run_train` / caminho_historico: {'entrypoint': 'run_train', 'check': 'caminho_historico', 'ok': True, 'error': None, 'retriever_is_sentinel': True}

- Grafo real com retriever resolvido pelo caminho oficial: {'scenario': 'entrypoint_caminho_oficial', 'call_log': ['diagnostic_abcd', 'diagnostic_menzies', 'diagnostic_spcl', 'summary'], 'error': None, 'ok': True, 'final_state_keys': ['diagnosis_abcd', 'diagnosis_menzies', 'diagnosis_spcl', 'image_data', 'image_path', 'lesion_size', 'seg_image_data', 'seg_path', 'status_abcd', 'status_menzies', 'status_spcl', 'validation'], 'final_report_present': False, 'validation_present': True, 'final_diagnostico_field': 'Lesão Benigna', 'summary_was_invoked': True, 'summary_prompt_marker_count': 3, 'diagnostic_coverage': True, 'pareceres_recebidos': 3, 'pareceres_esperados': 3, 'agentes_ausentes': [], 'consolidacao_parcial': False, 'status_final': 'determinado', 'n_calls_per_agent': {'diagnostic_spcl': 1, 'diagnostic_abcd': 1, 'diagnostic_menzies': 1, 'summary': 1}}


Nenhuma divergência encontrada.
