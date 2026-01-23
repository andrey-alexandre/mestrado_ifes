from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.llms import FakeListLLM
from app.config import settings

def get_llm(model_name: str, temperature: float = 0.7):
    if model_name.startswith("dummy"):
        # Default responses
        abcd_resp = "{\"assimetria\": \"0\", \"bordas\": \"0\", \"cor\": \"1\", \"estruturas\": \"1\", \"explicacao\": \"Dummy ABCD response\"}"
        menzies_resp = "{\"positivas\": {\"veu_azul_branco\": \"0\", \"multiplos_pontos_marrons\": \"0\", \"pseudopodes\": \"0\", \"streaming_radial\": \"0\", \"despigmentacao_tipo_cicatriz\": \"0\", \"pontos_pretos_perifericos\": \"0\", \"multiplas_cores\": \"0\", \"varios_pontos_saliencias_azul_acinzentados\": \"0\", \"linhas_ramificadas\": \"0\"}, \"negativas\": {\"simetria\": \"0\", \"cor_unica\": \"0\"}, \"explicacao\": \"Dummy Menzies response\"}"
        spcl_resp = "{\"padrao_de_pigmentacao_atipico\": \"0\", \"veu_azul_branco_irregular\": \"0\", \"vasos_atipicos\": \"0\", \"padrao_de_reticulo_irregular\": \"0\", \"padrao_de_globulos_irregulares\": \"0\", \"hiperpigmentacao_localizada\": \"0\", \"regressao\": \"0\", \"explicacao\": \"Dummy SPCL response\"}"
        summary_resp = "{\"diagnostico\": \"Lesão Benigna\", \"justificativa\": \"Dummy summary based on benign inputs\", \"recomendacoes\": \"Routine checkup\"}"

        if "abcd" in model_name:
            responses = [abcd_resp]
        elif "menzies" in model_name:
            responses = [menzies_resp]
        elif "spcl" in model_name:
            responses = [spcl_resp]
        elif "summary" in model_name:
            responses = [summary_resp]
        else:
            # Fallback for generic dummy
            responses = [abcd_resp]

        return FakeListLLM(responses=responses * 10)

    elif model_name.startswith("gpt"):
        return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=model_name, temperature=temperature)
    else:
        return ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=temperature)
