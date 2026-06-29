# Detecção de Phishing em URLs com Classificadores Clássicos

Trabalho final da disciplina de Reconhecimento de Padrões.

## Autores
- Vinícius Magalhães Dias
- João Marcello Machado Braz
- Pedro Gabryel Araujo do Nascimento

## Reprodução

```bash
pip install -r requirements.txt
python analise.py
pdflatex artigo.tex && bibtex artigo && pdflatex artigo.tex && pdflatex artigo.tex
```

O script `analise.py` gera todas as tabelas e figuras em `resultados/`,
que são consumidas pelo `artigo.tex` via `\input{}`.

## Estrutura

- `analise.py` — pipeline completo de análise.
- `artigo.tex` — fonte LaTeX (template SBC, 6 páginas).
- `referencias.bib` — bibliografia.
- `resultados/` — saídas do `analise.py`.
