from uncompyle6.main import decompile_file

with open('desenhou_decompilado.py', 'w', encoding='utf-8') as f:
    decompile_file('desenhou.cpython-313.pyc', f)
