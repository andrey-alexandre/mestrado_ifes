from langchain.schema import SystemMessage, HumanMessage

class DiagnosticAgent:
    def __init__(self, llm, content_txt):
        self.llm = llm
        self.content = content_txt

    def analyze_lesion(self, state):
      content = [
          # {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{state.image_data}'},
          {'type': 'text', 'text': "Aqui está a imagem original da lesão:"},
          {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{state.image_data}'},
          {'type': 'text', 'text': "Aqui está a versão segmentada da lesão:Aqui está a versão segmentada da lesão:"},
          {'type': 'image_url', 'image_url': f'data:image/jpeg;base64,{state.seg_image_data}'},
          {'type': 'text', 'text': self.content}
      ]
      msg = [HumanMessage(content=content)]
      return {"diagnosis_abcd": llm.invoke(msg).content}