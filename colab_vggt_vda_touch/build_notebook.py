"""Serialize the percent-format source into a self-contained Kaggle notebook."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'notebook_source.py').read_text()
cells = []
kind, lines = None, []

def flush():
    if kind is None:
        return
    text = '\n'.join(lines).strip() + '\n'
    if kind == 'markdown':
        text = '\n'.join(line[2:] if line.startswith('# ') else (line[1:] if line.startswith('#') else line)
                         for line in text.splitlines()) + '\n'
    else:
        ast.parse(text)
    cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True), id=f'cell-{len(cells):02d}')
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

for line in source.splitlines():
    if line.startswith('# %%'):
        flush()
        kind = 'markdown' if '[markdown]' in line else 'code'
        lines = []
    else:
        lines.append(line)
flush()
notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},
})
target = ROOT/'touch_vggt_vda_bytetrack_kaggle.ipynb'
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print(f'{target}: {len(cells)} cells, syntax checked')
