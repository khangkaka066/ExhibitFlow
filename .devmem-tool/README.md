# Portable devmem

Run this from the root of the target project:

```bash
python3 .devmem-tool/run.py init
python3 .devmem-tool/run.py session start "describe the task"
python3 .devmem-tool/run.py run -- pytest -q
python3 .devmem-tool/run.py session end
python3 .devmem-tool/run.py handoff
```

No package is installed and all records are stored in
`.devmem-tool/state/`. To remove the tool and all its data:

```bash
rm -rf .devmem-tool
```

`run` and `verify` execute only commands supplied explicitly after `--`.
