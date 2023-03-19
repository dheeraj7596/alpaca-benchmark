from accelerate import Accelerator
import transformers
import time

if __name__ == "__main__":
    accelerator = Accelerator()
    model = transformers.AutoModelForCausalLM.from_pretrained("/data/dheeraj/llama-7b")

    model = accelerator.prepare(model)
    time.sleep(180)
    pass
