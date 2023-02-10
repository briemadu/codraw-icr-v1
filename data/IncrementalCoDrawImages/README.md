# Prepare the raw JSON files

Clone the CoDraw repository

Then fragment the big JSON file into mini JSON files based on dialog

```
python script/preprocess.py dataset/CoDraw_1_0.json
```

There should be now a folder `output` with multiple json files.

# Convert the scene descriptions

The json files contain scenes described by the strings `abs_t`, `abs_b` and `abs_d`. The `abs_d` values contain the scenes created by the drawer after an interaction.

We convert these scene descriptions back into the SceneRenderer format.

```bash
python backvert.py output
```

This will result in a single file `DrawerScenes.txt` in the `output` directory.

# Create the drawed images

Clone the AbstractScenes_v1.1 repository

The repository should contain a `SceneRenderer.exe` and the `Pngs` directory.

Place the `DrawerScenes.txt` into the directory next to the SceneRenderer and execute

```
SceneRenderer.exe DrawerScenes.txt Pngs DrawerScenes
```

This will create a new directory `DrawerScenes` that contain the scenes as images.

Each of the images is named like `Scene3_1` where 3 is the `image_id` of the according JSON file e.g. `train_00003.json` and 1 is the position of the scene in the interaction (zero-index).


# Credits:

Thanks to Philipp Sadler who created this script and README.

Minor fix by us: backvert.py now checks validity using
```return self.x != "-10000" and self.y != "-10000"```.
We checked together that this had no effect in the scenes for CoDraw JSON.
