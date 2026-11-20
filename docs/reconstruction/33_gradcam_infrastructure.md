# 33 — Infraestrutura de Grad-CAM da CNN (40 imagens, classe-alvo melanoma)

Na branch `feature/inspection`. **É uma análise qualitativa dos resultados — não comprova raciocínio clínico correto nem causalidade, e não constitui critério de aprovação do modelo.** Nenhum peso, limiar ou pré-processamento foi alterado; nenhuma inferência da CNN fora do necessário para gerar os mapas foi feita; nenhum mapa foi enviado a LLMs.

## 1. Seleção (semente 42)

`src/cnn_baseline/gradcam_sampling.py` (pura, testável sem I/O) + `scripts/select_gradcam_sample.py` — 10 por célula de confusão (TP/TN/FP/FN) das predições S0 já existentes (`experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv`), usando o limiar já fixado (Youden, BCN20000). Candidatos ordenados por `lesion_id` antes do sorteio (reprodutibilidade independente da ordem do CSV). Célula com menos de 10 elegíveis usa todos e registra o atalho (`shortfall_groups`) — nenhum substituto buscado em outra célula. Saída: `experiments/evaluation/gradcam/gradcam_sample_manifest.csv` + `gradcam_sample_report.md`.

## 2. Geração dos mapas

`src/cnn_baseline/gradcam.py` (pura) + `scripts/generate_gradcam_gallery.py`.

- Checkpoint verificado contra o SHA-256 registrado (`42c385a36845bdfeb33c308862973dc55b1d92c0a1779bdf3193ebd9e6ede0c3`, doc 29) **antes** de qualquer uso — aborta (`SystemExit`) se não bater, nunca segue com um checkpoint não verificado.
- Classe-alvo: sempre melanoma — o único logit do classificador binário (`model.classifier[-1]` = `nn.Linear(in_features, 1)`; rótulo 1=melanoma na convenção de treinamento, `BINARY_TARGET_TO_LABEL`). Tensor-alvo documentado explicitamente no módulo: o logit pré-sigmoide desse único neurônio, o mesmo tensor cujo sigmoide já é usado pela avaliação S0.
- Camada-alvo: `model.features[-1]` (último `Conv2dNormActivation` da EfficientNetV2-S do torchvision, último mapa espacial antes do global-average-pool) — escolha padrão para Grad-CAM em EfficientNet, documentada, nenhuma outra camada testada/comparada.
- Pré-processamento idêntico ao de `scripts/run_cnn_s0_ham10000_eval.py` (`EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()`) — mesma função, não reimplementada.
- `model.eval()` (BatchNorm/Dropout em modo de inferência) com gradientes habilitados só para o backward do Grad-CAM (`torch.set_grad_enabled(True)`, nunca `torch.no_grad()`) — nenhum `optimizer.step()`; pesos nunca mudam (verificado em `tests/cnn_baseline/test_gradcam.py::test_model_weights_unchanged_after_gradcam`).
- Normalização min-max **por imagem**, independente entre imagens — documentado explicitamente que magnitude absoluta não é comparável entre mapas normalizados individualmente.
- Mapas vazios (`mapa_vazio_zero_apos_relu`), constantes não-zero (`mapa_constante_nao_zero`) e falhas (`falha`) recebem status explícito, aparecem na galeria sem sobreposição fictícia, nunca são substituídos silenciosamente.
- Sobreposição feita sobre a VISÃO PRÉ-PROCESSADA (mesmo resize/crop do pipeline oficial, desnormalizada de volta para RGB), não sobre a imagem original inteira — é essa visão que o modelo efetivamente recebe e à qual o mapa corresponde espacialmente; a imagem original é preservada separadamente, sem sobreposição.
- Registro por imagem (`gradcam_execution_log.csv`): lesão, checkpoint/hash, camada, classe-alvo, score, decisão, dispositivo, duração, status.
- Custo real registrado: tempo total e tempo médio por imagem, impressos e salvos — a amostra nunca é ampliada automaticamente (lote inicial de até 40).

Saída: `experiments/evaluation/gradcam/images/` (`_original.png`, `_input.png` [visão pré-processada], `_overlay.png`, `_cam.npy`), `gradcam_execution_log.csv`, `gradcam_gallery.html` (galeria por célula TP/TN/FP/FN).

## 3. Comandos para o pesquisador (nenhum executado nesta sessão)

```
python -m pytest tests/cnn_baseline/test_gradcam_sampling.py tests/cnn_baseline/test_gradcam.py -v

python scripts/select_gradcam_sample.py
python scripts/generate_gradcam_gallery.py \
    --checkpoint-path experiments/cnn_baseline/training/best_checkpoint.pt \
    --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000 \
    --device mps
```

Dependências: todas já existentes (`torch`, `torchvision`, `opencv-python-headless`, `numpy`, `Pillow`) — nenhuma nova.

## 4. Limites preservados

Nenhum peso, limiar ou pré-processamento foi alterado. Nenhuma inferência nova além da necessária para os 40 mapas. Nenhum mapa foi enviado a LLMs nesta etapa. A análise quantitativa em desenvolvimento não foi tocada.
