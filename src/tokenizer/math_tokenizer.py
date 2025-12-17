"""
Specialized Tokenizer for Mathematical Reasoning

This tokenizer handles both natural language and mathematical notation,
including LaTeX symbols and formulas common in olympiad-level mathematics.
"""

import re
import json
from typing import List, Dict, Optional, Union
from pathlib import Path


class MathTokenizer:
    """
    A specialized tokenizer for mathematical text.

    Features:
    - Handles LaTeX mathematical symbols (∀, ∃, ∑, ∫, etc.)
    - Preserves mathematical expressions
    - BPE-based encoding for natural language
    - Special tokens for mathematical structures
    """

    # Mathematical symbols and their LaTeX equivalents
    MATH_SYMBOLS = {
        # Greek letters (lowercase)
        "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta",
        "ε": "\\epsilon", "ζ": "\\zeta", "η": "\\eta", "θ": "\\theta",
        "ι": "\\iota", "κ": "\\kappa", "λ": "\\lambda", "μ": "\\mu",
        "ν": "\\nu", "ξ": "\\xi", "π": "\\pi", "ρ": "\\rho",
        "σ": "\\sigma", "τ": "\\tau", "υ": "\\upsilon", "φ": "\\phi",
        "χ": "\\chi", "ψ": "\\psi", "ω": "\\omega",

        # Greek letters (uppercase)
        "Γ": "\\Gamma", "Δ": "\\Delta", "Θ": "\\Theta", "Λ": "\\Lambda",
        "Ξ": "\\Xi", "Π": "\\Pi", "Σ": "\\Sigma", "Φ": "\\Phi",
        "Ψ": "\\Psi", "Ω": "\\Omega",

        # Blackboard bold (number sets)
        "ℕ": "\\mathbb{N}", "ℤ": "\\mathbb{Z}", "ℚ": "\\mathbb{Q}",
        "ℝ": "\\mathbb{R}", "ℂ": "\\mathbb{C}", "ℙ": "\\mathbb{P}",
        "𝔸": "\\mathbb{A}", "𝔹": "\\mathbb{B}", "𝔼": "\\mathbb{E}",
        "𝔽": "\\mathbb{F}", "𝔾": "\\mathbb{G}", "ℍ": "\\mathbb{H}",

        # Superscripts
        "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
        "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
        "⁺": "^+", "⁻": "^-", "⁼": "^=", "⁽": "^(", "⁾": "^)",
        "ⁿ": "^n", "ⁱ": "^i", "ˣ": "^x",

        # Subscripts
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
        "₊": "_+", "₋": "_-", "₌": "_=", "₍": "_(", "₎": "_)",
        "ₐ": "_a", "ₑ": "_e", "ₒ": "_o", "ₓ": "_x", "ₕ": "_h",
        "ₖ": "_k", "ₗ": "_l", "ₘ": "_m", "ₙ": "_n", "ₚ": "_p",
        "ₛ": "_s", "ₜ": "_t",

        # Mathematical operators
        "∀": "\\forall", "∃": "\\exists", "∄": "\\nexists",
        "∈": "\\in", "∉": "\\notin", "∋": "\\ni", "∌": "\\notni",
        "⊂": "\\subset", "⊃": "\\supset", "⊆": "\\subseteq", "⊇": "\\supseteq",
        "⊄": "\\nsubseteq", "⊅": "\\nsupseteq",
        "∪": "\\cup", "∩": "\\cap", "∅": "\\emptyset", "⊎": "\\uplus",
        "⊓": "\\sqcap", "⊔": "\\sqcup",

        # Relations
        "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "≈": "\\approx",
        "≡": "\\equiv", "∼": "\\sim", "≅": "\\cong", "≃": "\\simeq",
        "≺": "\\prec", "≻": "\\succ", "≼": "\\preceq", "≽": "\\succeq",
        "∣": "\\mid", "∤": "\\nmid", "⊢": "\\vdash", "⊨": "\\models",

        # Calculus
        "∫": "\\int", "∬": "\\iint", "∭": "\\iiint", "∮": "\\oint",
        "∑": "\\sum", "∏": "\\prod", "∐": "\\coprod",
        "∂": "\\partial", "∇": "\\nabla", "∆": "\\Delta", "∞": "\\infty",
        "√": "\\sqrt", "∛": "\\cbrt", "∜": "\\fourthroot",

        # Logic
        "∧": "\\land", "∨": "\\lor", "¬": "\\neg", "⊤": "\\top", "⊥": "\\bot",
        "⇒": "\\Rightarrow", "⇐": "\\Leftarrow", "⇔": "\\Leftrightarrow",
        "→": "\\to", "←": "\\from", "↔": "\\leftrightarrow",
        "↦": "\\mapsto", "⟼": "\\longmapsto",
        "⊨": "\\models", "⊢": "\\vdash", "⊣": "\\dashv",

        # Arrows
        "↑": "\\uparrow", "↓": "\\downarrow", "⇑": "\\Uparrow", "⇓": "\\Downarrow",
        "↗": "\\nearrow", "↘": "\\searrow", "↙": "\\swarrow", "↖": "\\nwarrow",
        "⇀": "\\rightharpoonup", "⇁": "\\rightharpoondown",
        "↼": "\\leftharpoonup", "↽": "\\leftharpoondown",

        # Set theory & misc
        "⊔": "\\sqcup", "⊓": "\\sqcap", "⊏": "\\sqsubset", "⊐": "\\sqsupset",
        "⊑": "\\sqsubseteq", "⊒": "\\sqsupseteq",
        "∝": "\\propto", "∥": "\\parallel", "⟂": "\\perp",
        "⊕": "\\oplus", "⊗": "\\otimes", "⊙": "\\odot", "⊖": "\\ominus",
        "⊘": "\\oslash", "⊞": "\\boxplus", "⊟": "\\boxminus",
        "⊠": "\\boxtimes", "⊡": "\\boxdot",

        # Dots and ellipsis
        "⋯": "\\cdots", "⋮": "\\vdots", "⋱": "\\ddots", "…": "\\ldots",
        "·": "\\cdot", "•": "\\bullet", "∘": "\\circ", "∙": "\\bullet",

        # Brackets and delimiters
        "⟨": "\\langle", "⟩": "\\rangle", "⟦": "\\llbracket", "⟧": "\\rrbracket",
        "⌈": "\\lceil", "⌉": "\\rceil", "⌊": "\\lfloor", "⌋": "\\rfloor",
        "｜": "\\vert", "∥": "\\Vert",

        # Geometry
        "∠": "\\angle", "∡": "\\measuredangle", "∢": "\\sphericalangle",
        "△": "\\triangle", "▵": "\\triangle", "□": "\\square",
        "◊": "\\diamond", "○": "\\circle", "●": "\\bullet",
        "°": "\\degree", "′": "\\prime", "″": "\\dprime",

        # Algebraic structures
        "⊗": "\\otimes", "⊕": "\\oplus", "⊞": "\\boxplus",
        "⋊": "\\rtimes", "⋉": "\\ltimes", "⋈": "\\bowtie",

        "\\frac": "\\frac",
        "\\boxed": "\\boxed",
        "\\text": "\\text",
        "\\cdot": "\\cdot",
        "\\ldots": "\\ldots",
        "\\sqrt": "\\sqrt",
        "\\sin": "\\sin",
        "\\cos": "\\cos",
        "\\tan": "\\tan",
        "\\log": "\\log",
        "\\ln": "\\ln",
        "\\lim": "\\lim",
        "^": "^",
        "_": "_",
        "{": "{",
        "}": "}",
        "(": "(",
        ")": ")",
        "[": "[",
        "]": "]",
        "+": "+",
        "-": "-",
        "=": "=",
    }

    # Special tokens
    SPECIAL_TOKENS = {
        "pad": "<pad>",
        "eos": "<eos>",
        "bos": "<bos>",
        "unk": "<unk>",
        "sep": "<sep>",
        # Mathematical structure tokens
        "math_start": "<math>",
        "math_end": "</math>",
        "equation_start": "<eq>",
        "equation_end": "</eq>",
        "proof_start": "<proof>",
        "proof_end": "</proof>",
        "solution_start": "<solution>",
        "solution_end": "</solution>",
        "step": "<step>",
        "step_end": "</step>",
        "answer_start": "<answer>",
        "answer_end": "</answer>",
    }

    def __init__(
        self,
        vocab_size: int = 50304,
        max_length: int = 8192,
    ):
        """
        Initialize the mathematical tokenizer.

        Args:
            vocab_size: Size of the vocabulary
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Build vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab()

        # Regex patterns for tokenization
        self._compile_patterns()

    def _build_vocab(self):
        """Build the initial vocabulary with special tokens and math symbols."""
        idx = 0

        # Add special tokens
        for token in self.SPECIAL_TOKENS.values():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Add mathematical symbols
        for symbol in self.MATH_SYMBOLS.keys():
            if symbol not in self.token_to_id:
                self.token_to_id[symbol] = idx
                self.id_to_token[idx] = symbol
                idx += 1

        # Add LaTeX commands
        for latex in self.MATH_SYMBOLS.values():
            if latex not in self.token_to_id:
                self.token_to_id[latex] = idx
                self.id_to_token[idx] = latex
                idx += 1

        # Add digits
        for i in range(10):
            token = str(i)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Add common ASCII characters
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Add common punctuation and operators
        for char in " .,;:!?()[]{}+-*/=<>^_|&%$#@~`'\"":
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # Store the base vocabulary size
        self.base_vocab_size = idx

    def _compile_patterns(self):
        """Compile regex patterns for tokenization."""
        # 1. Pattern to match Special Tokens (CRITICAL FIX)
        # We sort by length (descending) so <solution> is matched before <
        special_tokens_pattern = '|'.join(re.escape(s) for s in sorted(self.SPECIAL_TOKENS.values(), key=len, reverse=True))
        self.special_token_pattern = re.compile(f'({special_tokens_pattern})')

        # 2. Pattern to match LaTeX commands
        self.latex_pattern = re.compile(r'\\[a-zA-Z]+')

        # 3. Pattern to match mathematical symbols
        # Sort by length to ensure \geq matches before \ge if both exist
        sorted_symbols = sorted(self.MATH_SYMBOLS.keys(), key=len, reverse=True)
        math_symbols_pattern = '|'.join(re.escape(s) for s in sorted_symbols)
        self.math_symbol_pattern = re.compile(f'({math_symbols_pattern})')

        # 4. Pattern to match numbers (including decimals and scientific notation)
        self.number_pattern = re.compile(r'\d+\.?\d*(?:[eE][+-]?\d+)?')

        # 5. Pattern to match words
        self.word_pattern = re.compile(r'\b\w+\b')

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        tokens = []
        pos = 0

        while pos < len(text):
            # Skip whitespace
            if text[pos].isspace():
                tokens.append(' ')
                pos += 1
                continue

            # Try match special tokens first
            special_match = self.special_token_pattern.match(text, pos)
            if special_match:
                tokens.append(special_match.group())
                pos = special_match.end()
                continue

            # Try to match LaTeX command
            latex_match = self.latex_pattern.match(text, pos)
            if latex_match:
                tokens.append(latex_match.group())
                pos = latex_match.end()
                continue

            # Try to match mathematical symbol
            symbol_match = self.math_symbol_pattern.match(text, pos)
            if symbol_match:
                tokens.append(symbol_match.group())
                pos = symbol_match.end()
                continue

            # Try to match number
            number_match = self.number_pattern.match(text, pos)
            if number_match:
                tokens.append(number_match.group())
                pos = number_match.end()
                continue

            # Try to match word
            word_match = self.word_pattern.match(text, pos)
            if word_match:
                tokens.append(word_match.group())
                pos = word_match.end()
                continue

            # Single character
            tokens.append(text[pos])
            pos += 1

        return tokens

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Encode text to token ids.

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        max_length = max_length or self.max_length
        all_input_ids = []
        all_attention_masks = []

        for t in texts:
            # Tokenize
            tokens = self._tokenize_text(t)

            # Convert to ids
            input_ids = []

            # Add BOS token
            if add_special_tokens:
                input_ids.append(self.token_to_id[self.SPECIAL_TOKENS["bos"]])

            # Convert tokens to ids
            for token in tokens:
                if token in self.token_to_id:
                    input_ids.append(self.token_to_id[token])
                else:
                    # Handle unknown tokens with character-level encoding
                    for char in token:
                        if char in self.token_to_id:
                            input_ids.append(self.token_to_id[char])
                        else:
                            input_ids.append(self.token_to_id[self.SPECIAL_TOKENS["unk"]])

            # Add EOS token
            if add_special_tokens:
                input_ids.append(self.token_to_id[self.SPECIAL_TOKENS["eos"]])

            # Truncate if needed
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                if add_special_tokens:
                    input_ids[-1] = self.token_to_id[self.SPECIAL_TOKENS["eos"]]

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            # Pad if needed
            if padding:
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    pad_id = self.token_to_id[self.SPECIAL_TOKENS["pad"]]
                    input_ids.extend([pad_id] * pad_length)
                    attention_mask.extend([0] * pad_length)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # Return as single list if input was a string
        if isinstance(text, str):
            return {
                "input_ids": all_input_ids[0],
                "attention_mask": all_attention_masks[0],
            }
        else:
            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
            }

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token ids to text.

        Args:
            token_ids: Token ids or list of token id sequences
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text or list of texts
        """
        if not token_ids:
            return ""

        # Check if it's a single sequence or batch
        is_single = isinstance(token_ids[0], int)
        if is_single:
            token_ids = [token_ids]

        special_token_ids = set(self.token_to_id[t] for t in self.SPECIAL_TOKENS.values())
        results = []

        for ids in token_ids:
            tokens = []
            for token_id in ids:
                if skip_special_tokens and token_id in special_token_ids:
                    continue
                if token_id in self.id_to_token:
                    tokens.append(self.id_to_token[token_id])
                else:
                    tokens.append(self.SPECIAL_TOKENS["unk"])

            # Join tokens
            text = "".join(tokens)
            # Clean up extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            results.append(text)

        return results[0] if is_single else results

    def save(self, path: Union[str, Path]):
        """Save tokenizer configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
        }

        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MathTokenizer":
        """Load tokenizer from saved configuration."""
        path = Path(path)

        with open(path / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(
            vocab_size=config["vocab_size"],
            max_length=config["max_length"],
        )
        tokenizer.token_to_id = config["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in config["id_to_token"].items()}

        return tokenizer

    @property
    def pad_token_id(self) -> int:
        """Get padding token id."""
        return self.token_to_id[self.SPECIAL_TOKENS["pad"]]

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token id."""
        return self.token_to_id[self.SPECIAL_TOKENS["eos"]]

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token id."""
        return self.token_to_id[self.SPECIAL_TOKENS["bos"]]

    @property
    def unk_token_id(self) -> int:
        """Get unknown token id."""
        return self.token_to_id[self.SPECIAL_TOKENS["unk"]]


def test_tokenizer():
    """Test the mathematical tokenizer."""
    tokenizer = MathTokenizer()

    # Test cases
    test_texts = [
        "For all x in R, x^2 >= 0",
        "∀x ∈ ℝ, x² ≥ 0",
        "Prove that ∑_{i=1}^{n} i = n(n+1)/2",
        "The integral ∫_{0}^{∞} e^{-x} dx = 1",
        "Let f: ℝ → ℝ be a function such that f(x) = x² + 2x + 1",
    ]

    print("Testing Mathematical Tokenizer\n" + "="*50)
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded["input_ids"])
        print(f"\nOriginal: {text}")
        print(f"Tokens: {len(encoded['input_ids'])}")
        print(f"Decoded: {decoded}")

    return tokenizer


if __name__ == "__main__":
    test_tokenizer()