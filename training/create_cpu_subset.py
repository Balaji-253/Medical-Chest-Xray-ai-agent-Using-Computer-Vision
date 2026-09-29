from pathlib import Path
import shutil
import pandas as pd

ROOT = Path("data/raw/NIH_ChestXray14")
IMG_ROOT = ROOT / "images"
OUT = Path("data/processed/NIH_ChestXray14")

TRAIN_N = 2250
VAL_N = 750
TEST_N = 750

df = pd.read_csv(ROOT / "Data_Entry_2017_v2020.csv")

files = {p.name: p for p in IMG_ROOT.rglob("*.png")}
df["path"] = df["Image Index"].map(files)
matched = df[df["path"].notna()].copy()

print(f"Extracted PNGs: {len(files)}")
print(f"Metadata matches: {len(matched)}")
print(f"Patients represented: {matched['Patient ID'].nunique()}")

train_names = {
    x.strip()
    for x in open(ROOT / "train_val_list.txt", encoding="utf-8")
    if x.strip()
}

test_names = {
    x.strip()
    for x in open(ROOT / "test_list.txt", encoding="utf-8")
    if x.strip()
}

train_pool = matched[matched["Image Index"].isin(train_names)].copy()
test_pool = matched[matched["Image Index"].isin(test_names)].copy()

# Only patients represented in the extracted image archive.
patients = train_pool["Patient ID"].drop_duplicates().sample(
    frac=1,
    random_state=42
).tolist()

# Select validation patients first.
val_patients = set()

for patient in patients:
    if len(val_patients) >= max(1, int(train_pool["Patient ID"].nunique() * 0.20)):
        break
    val_patients.add(patient)

val_df = train_pool[train_pool["Patient ID"].isin(val_patients)].copy()
train_df = train_pool[~train_pool["Patient ID"].isin(val_patients)].copy()

# Deterministic image-level sampling within already patient-disjoint groups.
train_df = train_df.sample(
    n=min(TRAIN_N, len(train_df)),
    random_state=42
)

val_df = val_df.sample(
    n=min(VAL_N, len(val_df)),
    random_state=42
)

test_df = test_pool.sample(
    n=min(TEST_N, len(test_pool)),
    random_state=42
)

# Clean old processed subset.
for split in ["train", "val", "test"]:
    split_dir = OUT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for item in split_dir.iterdir():
        if item.is_file():
            item.unlink()

# Copy selected images and metadata.
for name, subset in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    out_dir = OUT / name

    for _, row in subset.iterrows():
        src = Path(row["path"])
        dst = out_dir / row["Image Index"]
        shutil.copy2(src, dst)

    subset[
        ["Image Index", "Patient ID", "Finding Labels"]
    ].to_csv(
        out_dir / "labels.csv",
        index=False
    )

    print(
        f"{name}: {len(subset)} images, "
        f"{subset['Patient ID'].nunique()} patients"
    )

# Verify patient separation.
train_ids = set(train_df["Patient ID"])
val_ids = set(val_df["Patient ID"])
test_ids = set(test_df["Patient ID"])

print("\nPatient overlap:")
print("Train ∩ Val:", len(train_ids & val_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Val ∩ Test:", len(val_ids & test_ids))

print("\nCPU subset created successfully.")
