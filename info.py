from datasets import load_dataset
from itertools import islice

ds = load_dataset(
    "AI4A-lab/RecruitView",
    split="train",
    streaming=True
)

rows = list(islice(ds, 10))

print("Sample keys:", rows[0].keys())
print("Sample question:", rows[0]["question"])
print("Sample confidence_score:", rows[0]["confidence_score"])
