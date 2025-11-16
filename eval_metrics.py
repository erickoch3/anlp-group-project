import evaluate


def bleu_func(preds, refs):
    bleu = evaluate.load("sacrebleu")
    return bleu.compute(predictions=preds, references=refs)["score"]


def chrf_func(preds, refs):
    chrf = evaluate.load("chrf")
    return chrf.compute(predictions=preds, references=refs)["score"]


def lenr_func(preds, refs):
    ratios = [len(p.split()) / len(r.split()) for p, r in zip(preds, refs)]
    return sum(ratios) / len(ratios)


def repr_func(preds, refs):
    rates = []
    for p in preds:
        toks = p.split()
        if len(toks) < 3:
            rates.append(0.0)
            continue
        bigrams = list(zip(toks, toks[1:]))
        u = len(set(bigrams)) / len(bigrams)
        rates.append(1 - u)
    return sum(rates) / len(rates)
