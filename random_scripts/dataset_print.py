from datasets import load_dataset
ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
print(ds)
print(ds[0])
print(ds[0]["answer"], type(ds[0]["answer"]))