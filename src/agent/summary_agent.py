class SummaryAgent:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, state):
      content_str=f"""Você está atuando como um especialista que recebeu três pareceres técnicos distintos sobre uma mesma lesão cutânea, baseados nos seguintes algoritmos:
      1️⃣ ABCD – {state.diagnosis_abcd}
      2️⃣ Menzies – {state.diagnosis_menzies}
      3️⃣ SPCL (Seven Point Checklist) – {state.diagnosis_spcl}

      Sua função é integrar os três pareceres fornecidos, comparando seus resultados e justificativas, para construir um **diagnóstico consolidado e fundamentado**.

      Considere:
      - O grau de concordância ou conflito entre os algoritmos;
      - A presença de padrões de alto risco recorrentes nos pareceres;
      - A confiabilidade dos critérios observados (por exemplo, múltiplas cores, bordas irregulares e estruturas atípicas);
      - A gravidade potencial com base em critérios combinados.

      🧠 Ao final, forneça um parecer clínico estruturado em formato JSON:

      ```json
      {{
        "Diagnóstico Final": "Suspeita de Melanoma / Lesão Benigna / Lesão Indeterminada",
        "Justificativa": "Integração das evidências dos algoritmos ABCD, Menzies e SPCL, destacando os principais achados.",
        "Recomendações": "Sugestões clínicas como biópsia, acompanhamento, ou nenhuma ação imediata."
      }}
      """
      content = [
          # {'type': 'text', 'text': "Aqui está a imagem original da lesão:"},
          # {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{state.image_data}'},
          # {'type': 'text', 'text': "Aqui está a versão segmentada da lesão:Aqui está a versão segmentada da lesão:"},
          # {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{state.seg_image_data}'},
          {'type': 'text', 'text': content_str}
      ]
      msg = [HumanMessage(content=content)]
      return {"validation": llm.invoke(msg).content}