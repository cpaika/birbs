# Linear A Decipherment Project

An attempt to decipher Linear A, the undeciphered writing system of the Minoan civilization (ca. 1800-1450 BCE) from Bronze Age Crete.

## Background

Linear A remains one of the great unsolved mysteries of ancient writing:
- ~1,400 inscriptions survive (compared to ~6,000 for Linear B)
- Many signs are shared with Linear B (which writes early Greek)
- We can assign phonetic values to many signs based on Linear B
- But the underlying **Minoan language** remains unknown

## Approach

This project takes a computational approach to decipherment:

1. **Corpus Analysis**: Statistical analysis of sign frequencies and patterns
2. **Phonetic Mapping**: Apply Linear B sound values to read Linear A
3. **Pattern Recognition**: Identify word boundaries, prefixes, suffixes
4. **Comparative Analysis**: Search for cognates in ancient Mediterranean languages
5. **Structural Analysis**: Analyze grammar through recurring patterns

## Project Structure

```
linear_a/
├── corpus/           # Linear A inscription data
├── signs/            # Sign inventory and phonetic values
├── analysis/         # Analysis tools and results
└── research/         # Research notes and findings
```

## Key Resources

- [SigLA Database](https://sigla.phis.me/) - Palaeographical database of Linear A signs
- [GORILA](https://en.wikipedia.org/wiki/Linear_A) - Godart & Olivier's inscription corpus
- Linear A Unicode range: U+10600–U+1077F

## Running the Analysis

```bash
python3 linear_a/analyze.py
```

## Current Status

🔬 **Active Research** - Attempting fresh decipherment from first principles
