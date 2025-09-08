from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def get_t5_pipeline(model_name="t5-3b", device=-1):
    """
    Create a HuggingFace pipeline for T5 model for text2text-generation.
    Args:
        model_name (str): Name or path of the T5 model.
        device (int): Device to run the pipeline on. -1 for CPU, >=0 for GPU.
    Returns:
        transformers.Pipeline: The text2text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    t5_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return t5_pipe

# Example usage:
t5_pipe = get_t5_pipeline("t5-3b", device=0)
result = t5_pipe("get the domains for this protein sequence: MCCLTSILPLAALAADAEKAPATTEAPAAEAPRPPLLERSQEDALALERLVPRAEQQTLQAGADSFLALWKPANDSDPQGAVIIVPGAGETADWPNAVGPLRQKFPDVGWHSLSLSLPDLLADSPQARVEAKPAAEPEKTKGESAPAKDVPADANANVAQATAADADTAESTDAEQASEQTDTADAERIFARLDAAVAFAQQHNARSIVLIGHGSGAYWAARYLSEKQPPHVQKLVMVAAQTPARVEHDLESLAPTLKVPTADIYYATRSQDRSAAQQRLQASKRQKDSQYRQLSLIAMPGNKAAEQEQLFRRVRGWMSPQG")
print(result)
