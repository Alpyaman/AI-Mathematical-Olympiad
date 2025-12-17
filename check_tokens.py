from src.tokenizer.math_tokenizer import MathTokenizer

tokenizer = MathTokenizer()
text = "<bos>\nProblem: x=1\n<solution>\n<step>"

tokens = tokenizer.encode(text, add_special_tokens=False)['input_ids']
decoded = tokenizer.decode(tokens, skip_special_tokens=False)

print(f"IDs: {tokens}")
print(f"Decoded: {decoded}")

# Check if <solution> is a single ID
sol_id = tokenizer.token_to_id["<solution>"]
if sol_id in tokens:
    print("✅ SUCCESS: <solution> is a single token!")
else:
    print("❌ FAILURE: <solution> was split!")