import pandas as pd


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""


if __name__ == "__main__":
    df = pd.read_csv("test.csv")

    new_prompts = []
    for p in list(df["prompt"]):
        temp = p.split("Text:")[0].strip().split("\n")
        new_temp = [w.strip() for w in temp]
        mod_p = " ".join(new_temp).strip()
        new_prompt = generate_prompt(mod_p)
        new_prompts.append(new_prompt)

    df["prompt"] = new_prompts
    df["text"] = new_prompts
    df.to_csv("./test_llama.csv", index=False)


