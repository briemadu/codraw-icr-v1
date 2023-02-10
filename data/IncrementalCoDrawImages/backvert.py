import json
import re
import os
import argparse
from tqdm import tqdm


class ClipArt:

    def __init__(self, d: str):
        parts = d.split(",")
        self.art_name = parts[0]
        # skip enumerate at parts[1]
        self.obj_idx = parts[2] # the json has swapped obj_idx and art_idx!
        self.art_idx = parts[3] # the json has swapped obj_idx and art_idx!
        self.x = str(int(float(parts[4])))
        self.y = str(int(float(parts[5])))
        self.z = parts[6]
        self.flip = parts[7]

    def is_valid(self):
        return self.x != "-10000" and self.y != "-10000"

    def __str__(self): # this complies to the README again (swap obj_idx and art_idx again)
        components = [self.art_name, self.art_idx, self.obj_idx, self.x, self.y, self.z, self.flip]
        return "\t".join(components)


class Scene:

    def __init__(self, dialog_idx, arts, verbose=True):
        self.dialog_idx = dialog_idx
        self.arts = arts
        self.verbose = verbose

    def __str__(self):
        lines = []
        valid_arts = [str(art) for art in self.arts if art.is_valid()]
        lines.append(f"{self.dialog_idx}\t{len(valid_arts)}")
        if self.verbose:
            lines.extend(valid_arts)
        return "\n".join(lines)


class Dialog:

    def __init__(self, dialog_idx):
        self.dialog_idx = dialog_idx
        self.scenes = []

    def add_description(self, scene_description):
        scene = convert_scene(self.dialog_idx, scene_description)
        self.scenes.append(scene)

    def __len__(self):
        return len(self.scenes)

    def __str__(self):
        if not self.scenes:
            return "<no scenes>"
        lines = [str(s) for s in self.scenes]
        return "\n".join(lines)


def convert_scene(dialog_idx, d):
    # empty strings will automatically result in a "default scene" image
    # find everything like 's_3s.png,0,3,0,465,35,0,1,'
    matches = re.findall('(\w+\.png[\d,-.]+)', d)
    arts = []
    for m in matches:
        try:
            arts.append(ClipArt(m))
        except Exception as e:
            print(m)
            print(e)
    scene = Scene(dialog_idx, arts)
    return scene


def convert_dialog(data):
    dialog_idx = data["image_id"]
    dialog = Dialog(dialog_idx)
    for step in data["dialog"]:
        # step_idx = step["seq_d"]
        image_d = step["abs_d"]
        dialog.add_description(image_d)
    return dialog


def convert_file(dir_name, file_name):
    with open(os.path.join(dir_name, file_name)) as f:
        data = json.load(f)
    return convert_dialog(data)


def convert_directory(dir_name, output_file="DrawerScenes.txt"):
    files = [f for f in os.listdir(dir_name) if f.endswith(".json")]
    dialogs = []
    for file in tqdm(files):
        dialogs.append(convert_file(dir_name, file))
    total_scenes = 0
    for d in dialogs:
        total_scenes += len(d)
    with open(os.path.join('./', output_file), "w") as out:
        out.write(f"{total_scenes}\n")
        for dialog in tqdm(dialogs):
            # print(len(dialog))
            # print(str(dialog))
            out.write(str(dialog))
            out.write("\n")


def main(source_directory):
    # convert_directory("../output")
    convert_directory(source_directory)


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert scene descriptions back from CoDraw to the original AbstractScenes format')
    parser.add_argument('directory', type=dir_path, #required=True,
                        help='Path to the prepared CoDraw JSON files directory')
    args = parser.parse_args()
    main(args.directory)
