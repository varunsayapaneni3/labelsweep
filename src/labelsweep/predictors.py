"""Model adapters. Each returns token-level spans typed with the label wording passed in."""

from __future__ import annotations

from bisect import bisect_right

from .data import Span


def join_tokens(tokens: list[str]) -> tuple[str, list[int], list[int]]:
    starts, ends, pos = [], [], 0
    for t in tokens:
        starts.append(pos)
        ends.append(pos + len(t))
        pos += len(t) + 1
    return " ".join(tokens), starts, ends


def char_to_token_spans(tokens: list[str], ents: list[dict]) -> list[Span]:
    """Map character-offset predictions on the space-joined text back to token spans.

    Offsets that fall inside a token snap outward to the containing token(s).
    """
    _, starts, _ = join_tokens(tokens)
    out: list[Span] = []
    seen = set()
    for e in ents:
        s_tok = max(bisect_right(starts, e["start"]) - 1, 0)
        e_tok = max(bisect_right(starts, max(e["end"] - 1, e["start"])) - 1, s_tok) + 1
        span = (s_tok, e_tok, e["label"])
        if span not in seen:
            seen.add(span)
            out.append(span)
    return out


class GlinerPredictor:
    """Zero-shot NER with a GLiNER checkpoint (pip install 'labelsweep[gliner]')."""

    def __init__(
        self,
        model: str,
        threshold: float = 0.5,
        batch_size: int = 16,
        device: str | None = None,
    ):
        import torch
        from gliner import GLiNER

        self.name = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model).to(self.device)
        self.model.eval()

    def predict(
        self, token_lists: list[list[str]], labels: list[str]
    ) -> list[list[Span]]:
        import torch

        texts = [join_tokens(t)[0] for t in token_lists]
        with torch.inference_mode():
            preds = self.model.inference(
                texts,
                labels,
                flat_ner=True,
                threshold=self.threshold,
                batch_size=self.batch_size,
            )
        return [
            char_to_token_spans(toks, ents)
            for toks, ents in zip(token_lists, preds, strict=True)
        ]
