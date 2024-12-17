import numpy as np

from model_loaders import load_pcw_wrapper
from logits_processor import RestrictiveTokensLogitsProcessor

from utils import encode_labels

wrapper = load_pcw_wrapper('meta-llama/Llama-2-7b-chat-hf', n_windows=2)#load_pcw_wrapper('gpt2-xl', n_windows=2)



output = wrapper.pcw_generate(contexts=["Review: Great movie! Sentiment: positive\n", "Review: Horrible film. Sentiment: negative\n"],
                              task_text="Classify the following example. Review: I don't liked it. Sentiment:",
                              temperature=1,
                              do_sample=False,
                              max_new_tokens=16)
print("============")
print(output)