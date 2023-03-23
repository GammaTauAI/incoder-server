from pathlib import Path
import argparse
import typing as t
import re
import subprocess
import sys
import traceback

from typing import List, Union

BOS = "<|endoftext|>"
EOM = "<|endofmask|>"

# NOTE: Does this have to be hard-coded?
MODEL_MAX_OUTPUT_LENGTH = 2048

_END_OF_MASK_OR_NEWLINE_RE = re.compile(r"<\|endofmask\|>|<\|endoftext\|>|\n", re.M)

# Regular expression that finds a function declaration in JS
# "function NAME(" or "function("
_FUNC_START_REGEX = re.compile(r"function(\s+([a-zA-Z_$][a-zA-Z_$0-9]*))?\s*\(")


def _extract_maybe_type(text: str) -> str:
    """
    Extracts the text before the end-of-mask, end-of-test, or newline tokens.
    Note that the result may be the empty string.
    """
    return _END_OF_MASK_OR_NEWLINE_RE.split(text, maxsplit=1)[0].strip()


def _templatize_function(line: str):
    """
    If the line contains function(x, y, z) then turn it into function(x ??, y ??, z ??)

    TODO: Support return type annotation
    """
    # Find the first occurrence of "function("
    function_start = _FUNC_START_REGEX.search(line)
    if function_start is None:
        return line
    function_start = function_start.end()
    # Find the first occurrence of ")"
    function_end = line.find(")", function_start)
    if function_end == -1 or function_end == function_start:
        return line
    arg_list = line[function_start:function_end].split(",")
    arg_list = "???, ".join(arg_list)
    return line[:function_start] + arg_list + "???" + line[function_end:]

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

def _clip_text(str1, str2, max_length):
    """
    Clips the two strings so that the total length is at most max_length.
    Keeps the first string intact, and clips the second string if possible
    """

    # Find the last occurrence of "function" in str1
    enclosing_function_start = str1.rfind("function")
    str1 = str1[enclosing_function_start:]

    if len(str1) < max_length:
        str2 = _prefix_ending_with_newline(str2, max_length - len(str1))
    elif len(str2) < max_length:
        # Negative, so we get the suffix
        str1 = _suffix_starting_with_newline(str1, max_length - len(str2))
    else:
        # Both exceed the max_length
        str1 = _suffix_starting_with_newline(str1, max_length // 2)
        str2 = _prefix_ending_with_newline(str2, max_length // 2)
    return str1, str2


class TypeInference:
    def __init__(self, model, tokenizer, temperature: float = 0.0, type_length_limit: int = 5, max_context_length: int = 70):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.type_length_limit = type_length_limit
        self.max_context_length = max_context_length
        self.do_sample = False if temperature == 0 else True
        self.type_log = []

    def _generate(
        self, prompt: str
    ) -> t.Tuple[str, bool]:
        """
        A canonical function to generate text from a prompt. The length_limit
        limits the maximum length of the generated text (beyond the prompt).
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        current_length = input_ids.flatten().size(0)
        max_length = self.type_length_limit + current_length

        if max_length == current_length:
            return prompt, True
        if max_length > MODEL_MAX_OUTPUT_LENGTH:
            max_length = MODEL_MAX_OUTPUT_LENGTH
            truncated = True
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
            prefix, suffix, self.max_context_length
        )
        prompt = f"{clipped_prefix}: <|mask:0|>{clipped_suffix}<|mask:1|><|mask:0|>"
        filled_type = self._generate_valid_type(
            prompt, retries=3
        )
        self.type_log.append({ "prompt": prompt, "type": filled_type })
        return filled_type

    def _infill_one(
            self,
            template: str,
            infill_marker: str,
            samples: int
        ) -> List[str]:
        parts = template.split(infill_marker, 1)
        if len(parts) < 2:
            raise ValueError(
                f"Expected at least one {infill_marker} in template. Got {template}"
            )
        infilled_prefix = parts[0]
        type_annotations: List[str] = []
        cur_sample_idx = 0
        while cur_sample_idx < samples:
            suffix = parts[1]
            type_annotation = self._get_type_annotation(infilled_prefix, suffix)
            type_annotations += [type_annotation]
            infilled_prefix += type_annotation
            cur_sample_idx += 1
        return type_annotations

    def _infill_many(
            self, template: str, infill_marker: str, samples: int
        ) -> List[List[str]]:
        parts = template.split(infill_marker)
        if len(parts) < 2:
            raise ValueError(
                f"Expected at least one {infill_marker} in template. Got {template}"
            )

        infilled_prefix = parts[0]
        type_annotations: List[List[str]] = []
        cur_sample_idx = 0
        while cur_sample_idx < samples:
            type_annotations_sample: List[str] = []
            for part_index, part in enumerate(parts[1:]):
                suffix = "".join(parts[part_index + 1 :])
                filled_type = self._get_type_annotation(infilled_prefix, suffix)
                type_annotations_sample += [filled_type]
                infilled_prefix += filled_type
            type_annotations += [type_annotations_sample]
            cur_sample_idx += 1
        return type_annotations

    # returns union now, must be fixed later
    def infer(
            self,
            code: str,
            samples: int,
            infer_single: bool
        ) -> Union[List[str], List[List[str]]]:
        self.type_log.clear()
        if "_hole_" not in code:
            return []
        if infer_single:
            return self._infill_one(code, infill_marker="_hole_", samples=samples)
        return self._infill_many(code, infill_marker="_hole_", samples=samples)

import model
m = model.init_model("facebook/incoder-6B")
typeinf = TypeInference(**model.init_model("facebook/incoder-6B"))

def split_string(string: str, max_length: int) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]

# use dumb split for characters * 4 = token approximation
def infill(code: str, samples: int, infill_single: bool, max_length: int = 2048):
    res = []
    for split_code in split_string(code, max_length * 4):
        res += [typeinf.infer(split_code, samples, infill_single)]
    return res

# for testing
def main() -> None:
    code = """function add(a: _hole_, b: _hole_): _hole_ {
    return a + b;
}"""
    print(infill(code, samples=3, infill_single=True))


if __name__ == "__main__":
    main()
