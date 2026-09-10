#!/bin/bash
for e in sd35_clean flux; do
  echo -n "$e : "
  /home/dell/Z1x/conda_envs/$e/bin/python -c 'import diffusers; print([n for n in dir(diffusers) if "Flux" in n][:6])' 2>&1 | tail -1
done
echo "=== FLUX model dirs ==="
ls -d /home/dell/models/FLUX* /home/dell/models/*flux* 2>/dev/null
find /home/dell/models -maxdepth 2 -name model_index.json 2>/dev/null | grep -i flux
echo "=== flux safetensors anywhere ==="
find /home/dell -maxdepth 3 -iname '*flux*.safetensors' 2>/dev/null | head -5
echo "=== DreamRenderer code? ==="
find /home/dell/Z1x -maxdepth 4 \( -iname '*dreamrenderer*' -o -iname '*dream_renderer*' \) 2>/dev/null | head
