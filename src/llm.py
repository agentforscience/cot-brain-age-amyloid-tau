"""Batched local-LLM inference engine (Qwen2.5-Instruct, HF transformers v5)."""
from __future__ import annotations
import os, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

MODELS = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "qwen1_5b": "Qwen/Qwen2.5-1.5B-Instruct"}

# Decoding configuration is FIXED across every arm, condition and model so that
# accuracy differences cannot be attributed to sampling. Greedy = deterministic.
GEN = dict(do_sample=False, temperature=None, top_p=None, top_k=None)


class Engine:
    def __init__(self, key: str, device: str = "cuda:0"):
        self.key, self.device = key, device
        name = MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    def chat(self, conversations, max_new_tokens=384, batch_size=32, prefill=None,
             desc="", verbose=True):
        """conversations: list[list[{role,content}]].
        prefill: optional list[str] of assistant-side text to force-continue from
        (used by the early-answering / mistake-injection faithfulness tests)."""
        n = len(conversations)
        flat = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                for c in conversations]
        if prefill is not None:
            flat = [t + p for t, p in zip(flat, prefill)]
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in flat]
        # length-sorted batching: groups similar-length prompts so padding (and therefore
        # both peak memory and wasted compute) is minimised. Order is restored afterwards.
        order = sorted(range(n), key=lambda i: lens[i])
        outs = [None] * n
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([flat[j] for j in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            try:
                with torch.no_grad():
                    o = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                            pad_token_id=self.tok.pad_token_id, **GEN)
                gen = o[:, enc["input_ids"].shape[1]:]
                dec = self.tok.batch_decode(gen, skip_special_tokens=True)
            except torch.OutOfMemoryError:      # fall back to one-at-a-time for this batch
                torch.cuda.empty_cache()
                dec = []
                for j in idx:
                    e1 = self.tok([flat[j]], return_tensors="pt",
                                  add_special_tokens=False).to(self.device)
                    with torch.no_grad():
                        o1 = self.model.generate(**e1, max_new_tokens=max_new_tokens,
                                                 pad_token_id=self.tok.pad_token_id, **GEN)
                    dec.append(self.tok.decode(o1[0, e1["input_ids"].shape[1]:],
                                               skip_special_tokens=True))
            for j, d in zip(idx, dec):
                outs[j] = d
            if verbose:
                print(f"    [{desc}] {min(i+batch_size, n)}/{n}", flush=True)
        return outs

    def close(self):
        del self.model
        gc.collect(); torch.cuda.empty_cache()
