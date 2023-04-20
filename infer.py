import re
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

    prefix_len = len(prefix)
    suffix_len = len(suffix)
    if prefix_len + suffix_len <= max_length:
        return prefix, suffix  # Nothing to do

    max_suffix_length = int(max_length / 2)
    max_prefix_length = max_length - max_suffix_length

    if prefix_len > max_prefix_length:
        prefix = prefix[-max_prefix_length:]

    if suffix_len > max_suffix_length:
        suffix = suffix[:max_suffix_length]

    return prefix, suffix


class TypeInference:
    def __init__(self, model, tokenizer, temperature: float = 0.0, type_length_limit: int = 5, max_length: int = 70, device = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.type_length_limit = type_length_limit
        self.max_length = max_length
        self.do_sample = False if temperature == 0 else True
        self.device = device

    def _generate(
        self, prompt: str
    ) -> Tuple[str, bool]:
        """
        A canonical function to generate text from a prompt. The length_limit
        limits the maximum length of the generated text (beyond the prompt).
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        current_length = input_ids.flatten().size(0)
        max_length = self.type_length_limit + current_length

        if max_length == current_length:
            return prompt, True
        else:
            truncated = False

        output = self.model.generate(
            input_ids=input_ids,
            do_sample=self.do_sample,
            top_p=0.95, # NOTE: Any point in this?
            temperature=self.temperature,
            max_length=max_length,
        )
        detok_hypo_str = self.tokenizer.decode(output.flatten())
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS) :]
        return detok_hypo_str, truncated

    def _generate_valid_type(
        self, prompt: str, retries: int
    ):
        """
        Given an InCoder-style prompt for infilling, tries to fill <|mask:0|> with a valid
        TypeScript type.
        """
        filled_type = "any"
        for _ in range(retries):
            generated, is_truncated = self._generate(prompt)
            if is_truncated:
                print("WARNING: Truncated output")
            filled_type = _extract_maybe_type(generated[len(prompt) :])
            if filled_type == "":
                filled_type = "any"
                continue
        return filled_type

    def _get_type_annotation(self, prefix: str, suffix: str) -> str:
        clipped_prefix, clipped_suffix = _clip_text(
            prefix, suffix, self.max_length
        )
        prompt = f"{clipped_prefix}: <|mask:0|>{clipped_suffix}<|mask:1|><|mask:0|>"
        filled_type = self._generate_valid_type(
            prompt, retries=3
        )
        return filled_type

    def _infill_one(self, template: str) -> str:
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

        type_annotation = self._get_type_annotation(clipped_prefix, clipped_suffix)
        return type_annotation

    # returns union now, must be fixed later
    def infer(self, code: str) -> Optional[str]:
        if INFILL_MARKER not in code:
            return ""
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
    type_annotations: List[str] = []
    while num_comps > 0:
        type_annotation = type_inf.infer(code)
        if type_annotation:
            type_annotations += [type_annotation]
        num_comps -= 1
    return type_annotations
