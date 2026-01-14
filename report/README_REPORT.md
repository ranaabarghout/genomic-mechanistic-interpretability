# Report Directory

This directory contains templates and resources for writing the final mechanistic interpretability report.

## Files

- **report_template.tex**: LaTeX template for 2-4 page report
- **references.bib**: BibTeX references for citations
- **README_REPORT.md**: This file (comprehensive guide)

## Quick Start

```bash
cd report/

# Compile with pdflatex
pdflatex report_template.tex
bibtex report_template
pdflatex report_template.tex
pdflatex report_template.tex

# View output
open report_template.pdf  # macOS
xdg-open report_template.pdf  # Linux
```

## Report Structure

The template follows standard scientific paper format:

1. **Abstract** (150-200 words)
   - Problem statement
   - Methods overview
   - Key findings
   - Implications

2. **Introduction** (~0.5 pages)
   - Motivation for interpretability
   - Research questions
   - Approach summary

3. **Methods** (~1.0 pages)
   - Model: DNABERT-2
   - Dataset: GTEx sQTL data
   - Four analyses:
     * Attention visualization
     * Activation patching
     * Circuit analysis
     * Sparse autoencoders

4. **Results** (~1.0 pages)
   - Attention patterns (Fig 1)
   - Causal components (Fig 2)
   - Functional circuits (Fig 3)
   - Interpretable features (Fig 4)

5. **Discussion** (~0.8 pages)
   - Biological interpretation
   - Comparison to prior work
   - Limitations
   - Future directions

6. **Conclusion** (~0.2 pages)
   - Summary of findings
   - Broader implications

## Creating Figures

Extract figures from your analysis outputs:

```bash
# Find your latest complete analysis
ls outputs/complete_analysis/

# Example: analysis_20260113_120000/
ANALYSIS_DIR="outputs/complete_analysis/analysis_20260113_120000"

# Copy key figures to report directory
cp $ANALYSIS_DIR/attention/*.png report/figures/
cp $ANALYSIS_DIR/activation_patching/*.png report/figures/
cp $ANALYSIS_DIR/circuit_analysis/*.png report/figures/
cp $ANALYSIS_DIR/sparse_autoencoder/*.png report/figures/
```

### Figure Guidelines

**Figure 1: Attention Patterns**
- Panel A: `attention_by_class_comparison.png`
- Panel B: `variant_position_attention.png`
- Panel C: Sample attention map (`sample_0_layer11_attention.png`)

**Figure 2: Activation Patching**
- Panel A: `1_layer_patching_effects.png`
- Panel B: `2_position_patching_effects.png`
- Panel C: `4_causal_traces.png`

**Figure 3: Circuit Analysis**
- Panel A: `1_discovered_circuits.png` (top-left panel)
- Panel B: `1_discovered_circuits.png` (bottom-left panel)
- Panel C: `4_circuit_ablation.png`

**Figure 4: SAE Features**
- Panel A: `2_feature_analysis.png` (top-left panel)
- Panel B: `2_feature_analysis.png` (top-right panel)
- Panel C: `feature_*_distribution.png` (pick top feature)

## Compiling the Report

### Option 1: Local LaTeX

```bash
# Install LaTeX (if needed)
# Ubuntu/Debian: sudo apt-get install texlive-full
# macOS: brew install basictex
# Windows: https://miktex.org/download

# Compile
cd report/
pdflatex report_template.tex
bibtex report_template
pdflatex report_template.tex
pdflatex report_template.tex
```

### Option 2: Overleaf (Recommended)

1. Go to [overleaf.com](https://www.overleaf.com)
2. Create new project → Upload Project
3. Upload `report_template.tex` and `references.bib`
4. Create `figures/` folder and upload your figures
5. Compile automatically

### Option 3: Docker

```bash
docker run --rm -v $(pwd):/workspace texlive/texlive \
    bash -c "cd /workspace/report && \
    pdflatex report_template.tex && \
    bibtex report_template && \
    pdflatex report_template.tex && \
    pdflatex report_template.tex"
```

## Filling in Results

Replace placeholder values in the template with your actual results:

### From Attention Analysis
Look in: `outputs/.../attention/attention_analysis_report.txt`

- Attention to variant position (fold change)
- Most important layers
- Top attention heads

### From Activation Patching
Look in: `outputs/.../activation_patching/activation_patching_report_*.txt`

- Most important layers (with effect sizes)
- Position importance statistics
- Top attention heads

### From Circuit Analysis
Look in: `outputs/.../circuit_analysis/circuit_analysis_report_*.txt`

- Number of circuits discovered
- Circuit differential activation values
- Ablation effects

### From SAE Analysis
Look in: `outputs/.../sparse_autoencoder/sae_analysis_report_*.txt`

- Number of features extracted
- Mean sparsity
- Number of differential features
- Top feature IDs

## Adding References

Add new citations to `references.bib`:

```bibtex
@article{newpaper2024,
  title={Title of Paper},
  author={Last, First and Last, First},
  journal={Journal Name},
  volume={XX},
  pages={YY--ZZ},
  year={2024}
}
```

Cite in text:
```latex
Previous work showed... \cite{newpaper2024}
```

## Submission Checklist

Before submitting your report:

- [ ] All results filled in from actual analysis outputs
- [ ] All 4 figures created and referenced
- [ ] Figure captions are descriptive
- [ ] Methods section has all parameters specified
- [ ] Statistical significance reported where appropriate
- [ ] Limitations section is complete
- [ ] References are complete
- [ ] PDF compiles without errors
- [ ] Page count is 2-4 pages
- [ ] Author name and email updated
- [ ] Repository link updated

## Tips for Writing

### Abstract
- **Sentence 1-2**: Problem and motivation
- **Sentence 3-4**: Methods overview
- **Sentence 5-7**: Key findings (quantitative)
- **Sentence 8**: Implications

Example:
> Genomic foundation models achieve state-of-the-art performance but lack mechanistic understanding. We applied four interpretability methods to DNABERT-2 for sQTL classification. We found that (1) attention concentrates 2.3× more on variant positions for significant sQTLs, (2) layers 8-11 are causally critical, (3) five functional circuits specialize in variant processing, and (4) SAE extracted 2048 interpretable features with 82% sparsity. These findings provide mechanistic insights into genomic variant processing.

### Results
- Start each subsection with main finding
- Use quantitative evidence (not just "increased")
- Reference figures: "Fig. 1A shows..."
- Include p-values for statistical tests

### Discussion
- Connect findings to biological knowledge
- Explain unexpected results
- Be specific about limitations
- Propose concrete next steps

## Example Workflow

```bash
# 1. Run complete analysis
python scripts/run_complete_analysis.py --num-samples 200 --train-sae

# 2. Find output directory
ANALYSIS_DIR=$(ls -td outputs/complete_analysis/analysis_* | head -1)
echo "Latest analysis: $ANALYSIS_DIR"

# 3. Copy figures
mkdir -p report/figures
cp $ANALYSIS_DIR/attention/attention_by_class_comparison.png report/figures/fig1a.png
cp $ANALYSIS_DIR/activation_patching/1_layer_patching_effects.png report/figures/fig2a.png
# ... etc

# 4. Extract key statistics
cat $ANALYSIS_DIR/attention/attention_analysis_report*.txt
cat $ANALYSIS_DIR/activation_patching/activation_patching_report*.txt
cat $ANALYSIS_DIR/circuit_analysis/circuit_analysis_report*.txt
cat $ANALYSIS_DIR/sparse_autoencoder/sae_analysis_report*.txt

# 5. Fill in template with actual numbers

# 6. Compile report
cd report/
pdflatex report_template.tex
bibtex report_template
pdflatex report_template.tex
pdflatex report_template.tex

# 7. Check output
open report_template.pdf
```

## Common Issues

**Issue**: `! LaTeX Error: File 'figures/fig1.pdf' not found`
**Solution**: Make sure figures are in `report/figures/` directory

**Issue**: Bibliography not appearing
**Solution**: Run `bibtex` then `pdflatex` twice more

**Issue**: Page count > 4 pages
**Solution**:
- Reduce figure sizes
- Condense discussion
- Move details to supplement

**Issue**: Figures too large/small
**Solution**: Adjust `\includegraphics[width=X\columnwidth]{...}` where X is 0.8-1.0

## Resources

### LaTeX
- [Overleaf Learn LaTeX](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [Detexify](http://detexify.kirelabs.org/classify.html) - Draw symbols to find LaTeX commands

### Scientific Writing
- [Nature: How to write a first-class paper](https://www.nature.com/articles/d41586-018-02404-4)
- [Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)
- [How to Write a Lot](https://www.apa.org/pubs/books/4441010) - Paul Silvia

### Interpretability
- [Anthropic: Transformer Circuits](https://transformer-circuits.pub/)
- [Neel Nanda: MI Explainer](https://www.neelnanda.io/mechanistic-interpretability/quickstart)
- [Circuits Thread](https://distill.pub/2020/circuits/)

## Contact

For questions about the report:
- Email: ranaabarghout@gmail.com
- GitHub Issues: https://github.com/ranaabarghout/genomic-mechanistic-interpretability/issues

## Original Report

Previous report by Rana Barghout:
`Barghout_Rana_Genomic_Interpretability.pdf`
