# -*- coding: utf-8 -*-
"""Extract prose-only plain text from matcom.tex for originality self-check."""
import re, sys, io, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(BASE, "matcom.tex"), encoding="utf-8").read()

# cut frontmatter/abstract region start and bibliography end
def cut(s, a, b=None):
    i = s.find(a)
    if i < 0:
        return s
    j = s.find(b, i) if b else len(s)
    return s[i:j if j > 0 else len(s)]

# We want: abstract + introduction + method prose + conclusion
parts = []
def grab(a, b):
    i = src.find(a)
    j = src.find(b, i + 1) if i >= 0 else -1
    if i >= 0:
        parts.append(src[i:j if j > 0 else len(src)])

grab("\\begin{abstract}", "\\end{abstract}")
grab("\\section{Introduction}", "\\section{Method}")
grab("\\section{Method}", "\\section{Truncation")
grab("\\section{Truncation", "\\section{Numerical examples}")
grab("\\section{Numerical examples}", "\\section{Conclusion}")
grab("\\section{Conclusion}", "\\section*{Data availability}")

text = "\n\n".join(parts)

# --- strip latex ---
# remove full environments that are math/table/figure
text = re.sub(r"\\begin\{(equation\*?|align\*?|multline\*?|gather\*?|eqnarray\*?|displaymath)\}.*?\\end\{\1\}", " ", text, flags=re.S)
text = re.sub(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", " ", text, flags=re.S)
text = re.sub(r"\\begin\{tikzcd\}.*?\\end\{tikzcd\}", " ", text, flags=re.S)
# inline math
text = re.sub(r"\\\((.*?)\\\)", " <MATH> ", text, flags=re.S)
text = re.sub(r"\$[^$]*\$", " <MATH> ", text)
# citations / refs
text = re.sub(r"\\cite[a-z]*\*?(\[[^\]]*\])*\{[^}]*\}", "", text)
text = re.sub(r"\\(ref|eqref|label)\{[^}]*\}", "", text)
text = re.sub(r"\\(emph|textit|textbf|texttt|textsc|mbox)\{([^{}]*)\}", r"\2", text)
text = re.sub(r"\\(section|subsection|paragraph|subsubsection)\*?\{([^{}]*)\}", r"\n\n\2\n\n", text)
# generic macro with one arg -> keep arg
text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
# remaining macros
text = re.sub(r"\\[a-zA-Z]+", " ", text)
text = text.replace("{", "").replace("}", "").replace("~", " ")
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
# collapse the repeated <MATH> filler
text = re.sub(r"( <MATH> ){2,}", " <MATH> ", text)
text = re.sub(r"(<MATH>[ ,.]*){2,}", "<MATH> ", text)
lines = [l.strip() for l in text.split("\n")]
lines = [l for l in lines if len(l) > 0]
text = "\n".join(lines)

out = os.path.join(BASE, "code", "_plaintext.txt")
open(out, "w", encoding="utf-8").write(text)
print("chars:", len(text))

# chunk into <=1900 chars on paragraph boundaries
CH = 1850
chunks = []
cur = ""
for p in text.split("\n"):
    p = p.strip()
    if not p:
        continue
    if len(cur) + len(p) + 1 > CH and cur:
        chunks.append(cur)
        cur = p
    else:
        cur = (cur + " " + p).strip()
if cur:
    chunks.append(cur)
for i, c in enumerate(chunks):
    open(os.path.join(BASE, "code", "_chunk_%02d.txt" % i), "w", encoding="utf-8").write(c)
print("chunks:", len(chunks), [len(c) for c in chunks])
