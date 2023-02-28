# Data

The following data/code directories are necessary:

- CoDraw-iCR (v1) Annotation
- AbstractScenes [link](https://www.microsoft.com/en-ca/download/details.aspx?id=52035)
- CoDraw: data code [link](https://github.com/facebookresearch/CoDraw)
- CoDraw: data file [link](https://drive.google.com/file/d/0B-u9nH58139bTy1XRFdqaVEzUGs/view?usp=sharing)
- CoDraw: model code [link](https://github.com/facebookresearch/codraw-models)
- Incremental CoDraw scenes


# Set Up

These are the steps to populate the ```data/``` directory. We do not provide a unified ```.sh``` script because some of them have to be downloaded manually.

### CoDraw-iCR (v1)

Our annotation is available at OSF: [https://osf.io/gcjhz/](https://osf.io/gcjhz/). You can download it manually or clone via the [osfclient](https://github.com/osfclient/osfclient).

### Abstract Scenes

We need the instruction giver's scenes which come from the AbstractScenes dataset. The dataset homepage is [here](https://www.microsoft.com/en-ca/download/details.aspx?id=52035). The download link below was copied from [this download page](https://www.microsoft.com/en-ca/download/confirmation.aspx?id=52035).

```bash
wget https://download.microsoft.com/download/4/5/D/45D1BBFC-7944-4AC5-AED2-1A35D85662D7/AbstractScenes_v1.1.zip
unzip AbstractScenes_v1.1.zip
rm -rf AbstractScenes_v1.1.zip
```

### CoDraw dataset: repository and data file

We need the CoDraw dataset repository, which contains code to breakdown the full JSON file into one file per dialogue, a step that is necessary to generate the step-by-step scenes by the instruction follower.

```bash
git clone https://github.com/facebookresearch/CoDraw.git
mv CoDraw CoDraw-master
cd CoDraw-master/dataset/
# this may not work and you'll need to upload it manually
gdown https://drive.google.com/uc?id=0B-u9nH58139bTy1XRFdqaVEzUGs -O CoDraw_1_0.json
cd ..
python script/preprocess.py dataset/CoDraw_1_0.json
cd ..
```

### Incremental CoDraw scenes

The code to generate the step-by-step scenes was kindly written by our colleague Philipp Sadler. It turns the dialogue JSON files into a .txt file that contains the scene string representations. To generate them, follow the instructions in ```IncrementalCoDrawImages/README```.


### CoDraw model code

We need some scripts from the [original CoDraw models](https://github.com/facebookresearch/codraw-models.git) to compute the scene similarity score. Please put the contents of [their code repository](https://github.com/facebookresearch/codraw-models/archive/refs/heads/master.zip ) into ```./codrawmodels```. The file ```codrawmodels/codraw_data.py``` needs to be manually updated by:

- replacing ```import abs_util_orig``` with ```from codrawmodels import abs_util_orig```
- set DATASET_PATH = Path('../data/CoDraw-master/dataset/CoDraw_1_0.json')
- Optional: comment out ```import abs_render```and the method definition ```def _repr_svg_(self)```.
