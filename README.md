# Timbre-Informed DDSP Violin
The corresponding master thesis is **Modelling Timbre for Neural Violin Synthesis**, which will be published in the [Zenodo](https://zenodo.org/communities/smc-master/records?q=&l=list&p=1&s=10&sort=newest) of Music Technology Group.

## Audio demos
[🎧 Live audio demos](https://kenzhuti.github.io/audio-demo/)

## Data preparation

### Violin Etudes

1. Download the Violin Etudes dataset with hmag and f0 labelling from [here](https://drive.google.com/file/d/1AbM6HxsajVcPC9oJLljAEEVqEldZwyKp/view?usp=sharing).
2. Extract the dataset to a directory, e.g., `data/violin_etudes_with_labelling`.
3. Run the following command to resample the dataset to 24 kHz wave files. The resampled files will be saved in the target directory with the same structure as the original files.
```bash
python scripts/resample_dir.py data/violin_etudes_raw data/violin_etudes --suffix .wav --sr 24000
```

### $f_0$ and hmag (harmonic magnitude) Extraction
1. SWIPE Method
<br>The f0s will be saved as `.pv` file in the same directory with the original files using 5 ms hop size.
```bash
python scripts/wav2f0.py data/violin_etudes_with_labelling
```

2. CHR Method
<br> Run `constrained_harmonic_resynthesis.py` and `timbre_clustering.ipynb` in the directory `constrained-harmonic-resynthesis4timbre-clustering` for the extraction of f0 and hmag labels. The f0s and hmags will be saved as `.tb.csv` file using sample-rate-wise interpolation for timestamps.

## Training

Below is the command to train each models.

```bash
python autoencode.py fit --config cfg/ae/violin.yaml --model cfg/ae/decoder/ddsp.yaml --trainer.logger false
```
To condition the DDSP model on hmag embedding and/or inharmonicity, modify in `violin.yaml` `model.init_args.use_hmag_embedding`, `model.init_args.use_inharmonicity`, and `data.init_args.use_hmag_embedding` to `true` or `false`. 

Alternatively, add `--model.init_args.use_hmag_embedding {boolean}`, `--model.init_args.use_inharmonicity {boolean}`, and `--data.init_args.use_hmag_embedding {boolean}` in the training command above, e.g, for model `ddsp-violin-hmag`.
```bash
python autoencode.py fit --config cfg/ae/violin.yaml --model cfg/ae/decoder/ddsp.yaml \
 --model.init_args.use_hmag_embedding true \
 --model.init_args.use_inharmonicity false \
 --data.init_args.use_hmag_embedding true \
 --trainer.logger false
```

By default, the checkpoints are automatically saved under `checkpoints/` directory. 
Feel free to remove `--trainer.logger false` and edit the logger settings in the configuration file `cfg/ae/violin.yaml` to fit your needs.
Please checkout the LightningCLI instructions [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

## Evaluation

### MCD/MSS

After training the models, you can evaluate the models using the following command. Replace `{YOUR_CONFIG}` and `{YOUR_CHECKPOINT}` with the corresponding configuration file and checkpoint.

```bash
python autoencode.py test -c {YOUR_CONFIG}.yaml --ckpt_path {YOUR_CHECKPOINT}.ckpt --data.duration 2 --data.overlap 0 --seed_everything false --data.wav_dir data/violin_etudes_with_labelling --data.batch_size 32 --trainer.logger false
```

### PESQ/FAD

For PESQ/FAD evaluation, you'll first need to store the synthesised waveforms in a directory. Replace `{YOUR_CONFIG}`, `{YOUR_CHECKPOINT}`, and `{YOUR_OUTPUT_DIR}` with the corresponding configuration file, checkpoint, and output directory.

```bash
python autoencode.py predict -c {YOUR_CONFIG}.yaml --ckpt_path {YOUR_CHECKPOINT}.ckpt --trainer.logger false --seed_everything false --data.wav_dir data/violin_etudes_with_labelling --trainer.callbacks+=ltng.cli.MyPredictionWriter --trainer.callbacks.output_dir {YOUR_OUTPUT_DIR}
```

Make a new directory and copy the test set indicated in `train_data/test_files.txt` from `violin_etudes_with_labelling`.

Then, calculate the PESQ scores:
    
```bash
python eval_pesq.py data/violin_etudes_test {YOUR_OUTPUT_DIR}
```

For the FAD scores:

```bash
python fad.py data/violin_etudes_test {YOUR_OUTPUT_DIR}
```

We use [fadtk](https://github.com/microsoft/fadtk) and [descript audio codec](https://github.com/descriptinc/descript-audio-codec) for the FAD evaluation. 


## Checkpoints

The checkpoints we used for evaluation are provided [here](ckpts).


## Citation

If this repository is useful to your research or project, please cite the following master’s thesis. The thesis is not yet published (expected October 2025). The template is provided below.

After publication (replace DOI once available on Zenodo)

- Plain text (APA/Chicago-style example):

```
Liu, Q. (2025, October). Modelling Timbre for Neural Violin Synthesis [Master's thesis]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

- BibTeX (with DOI, recommended @misc and Zenodo as publisher):

```bibtex
@misc{liu_thesis_2025_zenodo,
    author    = {Liu, Qin},
    title     = {Modelling Timbre for Neural Violin Synthesis},
    year      = {2025},
    publisher = {Zenodo},
    doi       = {10.5281/zenodo.XXXXXXX},
    url       = {https://doi.org/10.5281/zenodo.XXXXXXX},
    note      = {Master's thesis, Music Technology Group, Universitat Pompeu Fabra}
}
```
