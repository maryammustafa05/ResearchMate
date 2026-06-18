from main import read_pdf, chunk_text

text = read_pdf("papers/sign_lang.pdf")  # use your actual filename
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks[:5]):
    print(f"\n=== Chunk {i} ===")
    print(c[:300])