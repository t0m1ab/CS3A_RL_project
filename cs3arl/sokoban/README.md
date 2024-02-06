# Sokoban environment and visualization tools

### Directories:

* `surface/` - Contains image data to plot Sokoban maps.
* `boxoban-levels/` - Contains a subset of [DeepMind boxoban levels](https://github.com/google-deepmind/boxoban-levels).
* `my-sokoban-levels/` - Contains custom Sokoban levels.

### Files:
* `sokoban_env.py` - Define a Sokoban environment.
* `dataloaders.py` - Tools to load Sokoban levels from DeepMind data files or custom data files.
* `render_utils.py` - Tools to render Sokoban maps.
* `sokoban_ui.py` - Gradio UI to play Sokoban using the environment defined in **sokoban_env.py**
* `run_ui.sh` - Bash script used to launch the UI in **sokoban_ui.py**.


### Launch the Sokoban UI:
```bash
bash run_ui.sh
```