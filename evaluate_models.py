import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from rope_marian import RotaryMarianMTModel
from eval_metrics import bleu_func, chrf_func, lenr_func, repr_func
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_texts(ds):
    src = [ex["translation"]["de"] for ex in ds]
    refs = [ex["translation"]["en"] for ex in ds]
    return src, refs


def translate(model, tokenizer, texts, gen_cfg, device, bs=32):
    outs = []
    for i in tqdm(range(0, len(texts), bs)):
        batch = texts[i : i + bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, generation_config=gen_cfg, use_cache=True)
        outs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return outs


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() 
        else torch.device("cpu")
    )

    data_id = load_dataset("EdinburghNLP/europarl-de-en-mini", split="validation")
    data_ood = load_dataset("EdinburghNLP/europarl-de-en-mini", split="gen_val")
    id_src, id_refs = get_texts(data_id)
    ood_src, ood_refs = get_texts(data_ood)

    models = {
        "my-de-en-nmt": "my-de-en-nmt",
        "my-de-en-nmt_rot": "my-de-en-nmt_rot",
    }

    metrics = {
        "bleu": bleu_func,
        "chrf": chrf_func,
        "lenr": lenr_func,
        "repr": repr_func,
    }

    results = {}

    for name, path in models.items():
        tok = AutoTokenizer.from_pretrained(path)
        if name=="my-de-en-nmt_rot":
            mdl = RotaryMarianMTModel.from_pretrained(path).to(device)
        else:
            mdl = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        mdl.eval()
        gen_cfg = GenerationConfig.from_pretrained(path)

        id_preds = translate(mdl, tok, id_src, gen_cfg, device)
        ood_preds = translate(mdl, tok, ood_src, gen_cfg, device)

        results[(name, "id")] = {k: fn(id_preds, id_refs) for k, fn in metrics.items()}
        results[(name, "ood")] = {k: fn(ood_preds, ood_refs) for k, fn in metrics.items()}

    # Print results
    for (name, split), m in results.items():
        print(name, split, m)

    # Plot bar chart
    order = [
        ("my-de-en-nmt", "id"),
        ("my-de-en-nmt_rot", "id"),
        ("my-de-en-nmt", "ood"),
        ("my-de-en-nmt_rot", "ood"),
    ]
    metric_names = ["bleu", "chrf", "lenr", "repr"]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()  # separate scale for lenr/repr on the right

    x_base = list(range(len(metric_names)))
    width = 0.18
    group_gap = width * 0.8

    offsets = [
        -1.5 * width - group_gap / 2,
        -0.5 * width - group_gap / 2,
        0.5 * width + group_gap / 2,
        1.5 * width + group_gap / 2,
    ]

    left_metrics = {"bleu", "chrf"}
    right_metrics = {"lenr", "repr"}
    left_idx = [i for i, m in enumerate(metric_names) if m in left_metrics]
    right_idx = [i for i, m in enumerate(metric_names) if m in right_metrics]

    for j, key in enumerate(order):
        vals = [results[key][mn] for mn in metric_names]

        if left_idx:
            x_left = [x_base[i] + offsets[j] for i in left_idx]
            v_left = [vals[i] for i in left_idx]
            ax_left.bar(
                x_left,
                v_left,
                width=width,
                label=f"{key[0]}-{key[1]}",
            )

        if right_idx:
            x_right = [x_base[i] + offsets[j] for i in right_idx]
            v_right = [vals[i] for i in right_idx]
            ax_right.bar(x_right, v_right, width=width)

    ax_left.set_xticks(x_base)
    ax_left.set_xticklabels(metric_names)

    ax_left.set_ylabel("BLEU/chrF")
    ax_right.set_ylabel("LENR/REPR")
    ax_left.set_title("Eval metrics on ID and OOD")

    handles, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(handles, labels, loc="best")

    fig.tight_layout()
    fig.savefig("eval_barplot.png", dpi=150)


if __name__ == "__main__":
    main()
