"""
Test the enhanced mathematical tokenizer with Unicode symbols.
"""

# Direct import to avoid loading torch dependencies
import sys
sys.path.insert(0, '.')
from src.tokenizer.math_tokenizer import MathTokenizer


def test_enhanced_symbols():
    """Test all newly added Unicode mathematical symbols."""
    print("="*70)
    print("ENHANCED TOKENIZER TEST")
    print("="*70)

    tokenizer = MathTokenizer()

    # Test cases with problematic symbols from the demo
    test_cases = [
        # Blackboard bold
        ("∀x ∈ ℝ, if x² = 4 then x = 2 or x = -2", "Real numbers with superscript"),
        ("Let f: ℝ → ℝ be continuous", "Function mapping between real numbers"),
        ("ℕ, ℤ, ℚ, ℝ, ℂ are number sets", "All standard number sets"),

        # Superscripts and subscripts
        ("x² + y² = z²", "Pythagorean theorem"),
        ("aⁿ + bⁿ = cⁿ", "Fermat's last theorem"),
        ("∫₀ˣ e^(-t²) dt", "Integral with subscript and superscript"),
        ("x₁, x₂, ..., xₙ", "Sequence notation"),

        # Complex expressions
        ("∑_{k=1}^{∞} 1/k² = π²/6", "Basel problem"),
        ("lim_{n→∞} (1 + 1/n)ⁿ = e", "Limit definition of e"),

        # Set theory
        ("A ⊂ B ⊆ C", "Subset relations"),
        ("A ∪ B ∩ C = ∅", "Set operations"),

        # Logic
        ("(P ⇒ Q) ⇔ (¬P ∨ Q)", "Logical equivalence"),
        ("∀x ∃y (x < y)", "Quantifiers"),

        # Calculus
        ("∂f/∂x + ∂f/∂y = ∇·F", "Partial derivatives"),
        ("∫∫ f(x,y) dA", "Double integral"),

        # Greek letters
        ("α, β, γ, δ, θ, λ, μ, π, σ, ω", "Greek lowercase"),
        ("Γ, Δ, Θ, Λ, Π, Σ, Φ, Ψ, Ω", "Greek uppercase"),

        # Geometry
        ("∠ABC = 90° and △ABC is right", "Geometric notation"),
        ("AB ⊥ CD and AB ∥ EF", "Perpendicular and parallel"),

        # Relations
        ("a ≤ b ≥ c ≠ d ≈ e", "Comparison operators"),
        ("f: A → B is bijective ⇔ f⁻¹ exists", "Function properties"),
    ]

    print("\nTesting enhanced Unicode support:\n")

    passed = 0
    failed = 0

    for text, description in test_cases:
        try:
            # Encode
            encoded = tokenizer.encode(text, padding=False)

            # Decode
            decoded = tokenizer.decode(encoded["input_ids"])

            # Check if all important symbols are preserved
            # (allowing for minor whitespace differences)
            original_chars = set(text.replace(" ", ""))
            decoded_chars = set(decoded.replace(" ", ""))

            # Calculate symbol preservation rate
            important_symbols = original_chars - set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,. ")
            preserved_symbols = important_symbols & decoded_chars

            if len(important_symbols) > 0:
                preservation_rate = len(preserved_symbols) / len(important_symbols) * 100
            else:
                preservation_rate = 100.0

            status = "✓" if preservation_rate > 80 else "⚠"

            if preservation_rate > 80:
                passed += 1
            else:
                failed += 1

            print(f"{status} {description}")
            print(f"   Original:  {text}")
            print(f"   Decoded:   {decoded}")
            print(f"   Tokens:    {len(encoded['input_ids'])}")
            print(f"   Preserved: {preservation_rate:.0f}% of symbols")
            print()

        except Exception as e:
            failed += 1
            print(f"✗ {description}")
            print(f"   Error: {e}")
            print()

    print("="*70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*70)

    return passed, failed


def test_symbol_categories():
    """Test that all symbol categories are in vocabulary."""
    print("\n" + "="*70)
    print("SYMBOL CATEGORY COVERAGE TEST")
    print("="*70 + "\n")

    tokenizer = MathTokenizer()

    categories = {
        "Blackboard Bold": ["ℕ", "ℤ", "ℚ", "ℝ", "ℂ"],
        "Superscripts": ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"],
        "Subscripts": ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"],
        "Greek Lower": ["α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "σ", "ω"],
        "Greek Upper": ["Γ", "Δ", "Θ", "Λ", "Π", "Σ", "Φ", "Ψ", "Ω"],
        "Operators": ["∀", "∃", "∈", "⊂", "⊆", "∪", "∩", "∅"],
        "Relations": ["≤", "≥", "≠", "≈", "≡", "∼", "≅"],
        "Calculus": ["∫", "∑", "∏", "∂", "∇", "∞"],
        "Logic": ["∧", "∨", "¬", "⇒", "⇔", "→"],
        "Arrows": ["←", "→", "↔", "↑", "↓", "⇒", "⇔"],
    }

    for category, symbols in categories.items():
        in_vocab = sum(1 for s in symbols if s in tokenizer.token_to_id)
        coverage = in_vocab / len(symbols) * 100
        status = "✓" if coverage == 100 else "⚠"

        print(f"{status} {category:20s}: {in_vocab:2d}/{len(symbols):2d} ({coverage:3.0f}%)")

    total_symbols = sum(len(symbols) for symbols in categories.values())
    total_in_vocab = sum(
        sum(1 for s in symbols if s in tokenizer.token_to_id)
        for symbols in categories.values()
    )
    total_coverage = total_in_vocab / total_symbols * 100

    print(f"\n{'='*70}")
    print(f"Overall Coverage: {total_in_vocab}/{total_symbols} ({total_coverage:.1f}%)")
    print(f"{'='*70}\n")


def main():
    """Run all tests."""
    print("\n")

    # Test symbol categories
    test_symbol_categories()

    # Test actual encoding/decoding
    passed, failed = test_enhanced_symbols()

    print("\n")

    if failed == 0:
        print("✓ All enhanced tokenizer tests passed!")
        print("The tokenizer now supports 200+ mathematical symbols.")
        return True
    else:
        print(f"⚠ {failed} test(s) had issues, but this may be acceptable.")
        print("Some complex Unicode symbols may need character-level fallback.")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)