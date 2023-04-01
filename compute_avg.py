import os

if __name__ == "__main__":
    dic = {
        "stem": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
                 "college_computer_science", "college_mathematics", "college_physics", "computer_security",
                 "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
                 "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
                 "high_school_physics", "high_school_statistics", "machine_learning"],
        "other": ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging",
                  "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
                  "professional_accounting", "professional_medicine", "virology"],
        "social sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
                            "human_sexuality", "professional_psychology", "public_relations", "security_studies",
                            "sociology", "us_foreign_policy"],
        "humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                       "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
                       "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
                       "world_religions"]
    }

    base_path = "/home/dheeraj/alpaca-benchmark/output/mmlu"
    total = 0
    acc = 0
    for key in dic:
        print(key)
        temp_acc = 0
        temp_total = 0

        for topic in dic[key]:
            filepath = os.path.join(base_path, topic, "out.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
            acc = float(lines[0].strip().split()[-1])
            docs = int(lines[1].strip().split()[-1])

            temp_acc += acc * docs
            temp_total += docs

        total += temp_total
        acc += temp_acc

        print("Avg acc", temp_acc / temp_total)
        print("*" * 80)

    print("Overall Avg acc", acc / total)
    print("*" * 80)
