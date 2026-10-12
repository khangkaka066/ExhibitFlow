"""Build the standalone Colab notebook for normal pose-motion IF training."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=(ROOT/'train_normal_motion_if_notebook.py').read_text()
cells=[]; kind=None; lines=[]

def flush():
    if kind is None:return
    text='\n'.join(lines).strip()+'\n'
    if kind=='markdown':
        text='\n'.join(line[2:] if line.startswith('# ') else (line[1:] if line.startswith('#') else line)
                       for line in text.splitlines())+'\n'
    else: ast.parse(text)
    cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True),
          'id':f'cell-{len(cells):02d}'}
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)

for line in source.splitlines():
    if line.startswith('# %%'):
        flush(); kind='markdown' if '[markdown]' in line else 'code'; lines=[]
    else: lines.append(line)
flush()
notebook={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'accelerator':'GPU','colab':{'name':'train_normal_pose_motion_isolation_forest.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'}}}
target=ROOT/'train_normal_pose_motion_isolation_forest.ipynb'
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print(f'{target}: {len(cells)} cells, syntax checked')
