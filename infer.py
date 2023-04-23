import re
import torch
from transformers import AutoTokenizer

from typing import List, Tuple, Optional

BOS = "<|endoftext|>"
EOM = "<|endofmask|>"

_END_OF_MASK_OR_NEWLINE_RE = re.compile(r"<\|endofmask\|>|<\|endoftext\|>|\n", re.M)
INFILL_MARKER = "_hole_"


def _extract_maybe_type(text: str) -> str:
    """
    Extracts the text before the end-of-mask, end-of-test, or newline tokens.
    Note that the result may be the empty string.
    """
    return _END_OF_MASK_OR_NEWLINE_RE.split(text, maxsplit=1)[0].strip()


def _prefix_ending_with_newline(str, max_length):
    """
    Produces a prefix of str is at most max_length, but does not split a line.
    """
    return str[:max_length].rsplit("\n", 1)[0]

def _suffix_starting_with_newline(str, max_length):
    """
    Produces a suffix of str is at most max_length, but does not split a line.
    """
    return str[-max_length:].split("\n", 1)[0]

def _clip_text(prefix, suffix, max_length):
    """
    Clip the prefix and suffix to be at most `max_length` characters long.
    The start of the prefix should be clipped, and the end of the suffix
    should be clipped. If both already fit within `max_length`, then do
    nothing.
    """
    # we need at least 2 characters to show something
    assert max_length >= 2

    prefix_len = len(prefix)
    suffix_len = len(suffix)

    if prefix_len + suffix_len <= max_length:
        return prefix, suffix  # nothing to do

    # distribute 3/4 of the max length to the prefix and 1/4 to the suffix
    prefix_max = int(max_length * 0.75)
    suffix_max = max_length - prefix_max

    # remember: we want to clip the start of the prefix and the end of the suffix
    # also, if we have leftover of the prefix, we want to give it to the suffix
    prefix_len = min(prefix_max, prefix_len)
    suffix_len = min(suffix_max, suffix_len)

    # if we have leftover of the prefix, we want to give it to the suffix
    leftover = max_length - prefix_len - suffix_len
    suffix_len += leftover

    return prefix[-prefix_len:], suffix[:suffix_len]


class TypeInference:
    def __init__(self, model, tokenizer, temperature: float = 0.0, type_length_limit: int = 5, max_length: int = 70, device = 0, num_comps: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.type_length_limit = type_length_limit
        self.max_length = max_length
        self.do_sample = False if temperature == 0 else True
        self.device = device
        self.num_comps = num_comps

    def _generate(self, prefix_suffix_tuples) -> List[str]:
        """
        A canonical function to generate text from a prompt. The length_limit
        limits the maximum length of the generated text (beyond the prompt).
        """
        if type(prefix_suffix_tuples) == tuple:
            prefix_suffix_tuples = [prefix_suffix_tuples]

        print(f"prefix_suffix_tuples: {prefix_suffix_tuples}")

        prompts= [f"{p}: <|mask:0|>{s}<|mask:1|><|mask:0|>" for p, s in prefix_suffix_tuples]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(self.device)

        max_length = inputs.input_ids[0].size(0) + self.max_length
        print(max_length)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.95,
                temperature=self.temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )

        detok_hypo_strs = [self.tokenizer.decode(output.flatten()) for output in outputs]
        detok_hypo_strs = [string[len(BOS) :] for string in detok_hypo_strs if string.startswith(BOS)]
        return detok_hypo_strs

    def _generate_valid_types(self, prefix: str, suffix: str, retries: int) -> List[str]:
        """
        Given an InCoder-style prompt for infilling, tries to fill <|mask:0|> with a valid
        TypeScript type.
        """
        for _ in range(retries):
            # generated = self._generate([prompt] * self.num_samples)
            generated = self._generate([(prefix, suffix)] * self.num_comps)

            checked_not_empty = []
            for g in generated:
                if g.strip() != "":
                    checked_not_empty.append(g.strip())

            if len(checked_not_empty) == 0:
                continue
            
            # filled_type = _extract_maybe_type(generated[len(prompt) :])
            # if filled_type == "":
                # filled_type = "any"
                # continue
            return checked_not_empty
        return ["any"]

    # def _get_type_annotation(self, prefix: str, suffix: str) -> List[str]:
        # clipped_prefix, clipped_suffix = _clip_text(
            # prefix, suffix, self.max_length
        # )
        # prompt = f"{clipped_prefix}: <|mask:0|>{clipped_suffix}<|mask:1|><|mask:0|>"
        # filled_types = self._generate_valid_types(
            # clipped_prefix, clipped_suffix, retries=3
        # )
        # return filled_types

    def _infill_one(self, template: str) -> List[str]:
        parts = template.split(INFILL_MARKER, 1)
        print(parts)
        if len(parts) < 2:
            raise ValueError(
                f"Expected at least one {INFILL_MARKER} in template. Got {template}"
            )
        infilled_prefix = parts[0]
        suffix = parts[1].replace(": " + INFILL_MARKER, "")

        print(f"\tleft:\n {infilled_prefix}\n\tright:\n {suffix}")

        clipped_prefix, clipped_suffix = _clip_text(
            infilled_prefix, suffix, self.max_length
        )

        print(
            f"\tclipped left:\n {clipped_prefix}\n\tclipped right:\n {clipped_suffix}")

        # type_annotations = self._get_type_annotation(clipped_prefix, clipped_suffix)
        # return type_annotations
        filled_types = self._generate_valid_types(clipped_prefix, clipped_suffix, retries=3)
        return filled_types

    # returns union now, must be fixed later
    def infer(self, code: str) -> List[str]:
        if INFILL_MARKER not in code:
            return []
        return self._infill_one(code)

def split_string(string: str, max_length: int) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]

tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")

def infer(model_dict: dict, code: str, num_comps: int, max_length: int = 2048, temperature: float = 1.0) -> List[str]:
    assert num_comps > 0 
    type_inf = TypeInference(
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        temperature=temperature,
        device=model_dict["device"],
        max_length=max_length,
    )
    return type_inf.infer(code)
