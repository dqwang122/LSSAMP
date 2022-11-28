# Latent Sequence-Structure Model for Antimicrobial Peptide (LSSAMP)

This code is for the paper 'Accelerating Antimicrobial Peptide Discovery with Latent Sequence-Structure Model'

Our code depends on:
* python==3.7
* torch==1.9.0

To install, use ` pip install -r requirements.txt `.

Some code are borrowed from [OpenNMT](https://opennmt.net/), [VQ-VAE](https://github.com/nadavbh12/VQ-VAE) and [PreSumm](https://github.com/nlpyang/PreSumm). Thanks for their great work!

The file organization is as follows:

```
antibody
├── configs                           # configs
├── data                             
│   └── vocab
├── src                                 
│   ├── models                        # LSSAMP
│   │   ├── generation                  # generation tools  
│   │   ├── vqvae                       # vqvae           
│   │   ├── conv.py                     # cnn for feature selection
|   │   ├── data_loader.py                
|   │   ├── data_utils.py
|   │   ├── decoder.py          
|   │   ├── encoder.py
|   │   ├── loss.py
|   │   ├── model_builder.py            # main models
|   │   ├── neural.py                   # modules
|   │   ├── optimizers.py
|   │   ├── predictor.py                # generator
|   │   ├── reporter.py                 # log tensorboard
|   │   ├── tokenizer.py                  
|   │   └── trainer.py    
│   ├── others                        # tools
│   ├── prepro                        # data preprocess
│   ├── scripts                       # analysis
│   │   ├── ampgenHelper.py             # helper for ampgen dataset
|   │   ├── calProperty.py              # computational attributes
|   │   ├── draw.py                     # draw script
|   │   └── uniprotHelper.py            # helper for uniprot dataset
│   ├── distributed.py                # code for distributed training
│   ├── post_stats.py                 # post-process
│   ├── preprocess.py                 # data preprocess entry
│   ├── train_lm.py                   # train lm for codebook index
│   └── train_vae.py                  # train vae
├── baseline.sh                   # evaluate generation results for baseline
├── finetune.sh                   # finetune model on AMP dataset
├── run.sh                        # train model on Protein dataset        
├── sample.sh                     # evaluate sampling examples from LSSAMP                                       
└── README.md                     # Readme
```  



## Download and Preprocess Datasets

We use [Uniprot](https://www.uniprot.org) as the protein dataset and [APD3](https://aps.unmc.edu) as the AMP dataset.
[Prospr](https://github.com/dellacortelab/prospr/tree/prospr1) is used to predict the secondary structure for peptide sequences.
Please download these data and use *src/preprocess.py* to convert them into *.pt* format.
The preprocessing command may look like this:

```shell
python src/preprocess.py -mode split_dataset_to_txt -raw_path ../raw_data/uniprot_all.fasta -save_path ../raw_data/uniprot_all
python src/preprocess.py -mode do_format_to_pt -suffix txt -raw_path ../raw_data/uniprot_all -save_path ../data/uniprot_all -vocab_path data/vocab

```

You may find *src/scripts/uniprotHelper.py* useful to read and transform fasta files.

Finally, the plain-format input should look like this:
```text
GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV
YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY
```

And the json-format input will be:
```json
{"seq": "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV", "ss": [8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8]}
{"seq": "YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY", "ss": [8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]}
```


## Training on Protein Datasets and AMP Datasets

**[Attention] Please modify the default data/working path before using these scripts.**

After preprocessing the dataset and putting them under the *data* directory, we can train the model from *run.sh*.
We provide the HDFS-based training scripts, and you can also delete all HDFS-related commands and just run from local path.
We use *.yaml* files to organize the parameters and put the examples under *configs* directory. Please double check these options and change as you like.

To start, just use:

```shell
bash run.sh <config-name> <save-name>
```

For further pre-training or finetuning on AMP datasets:
```shell
bash finetune.sh <config-name> <expr-name> <load-model-name>
```

## Evaluation on baselines and LSSAMP

We implement several computational metrics in *src/scripts/calProperty.py*. The *baseline.sh* and *sample.sh* are provided for evaluating from baseline outputs and our generation results. To use them:
```shell
bash baseline.sh <model-name> <expr-name>
bash sample.sh <expr-name> <best-epoch> <best-step> <sample-method>
```
Pay attention that we need to predict the secondary structure labels for baseline outputs since they do not provide this information during the generation. For simplicity, we use our best LSSAMP model to predict the secondary structure labels (*configs/predict_ss.yaml*). One can also use Prospr to predict the secondary structure, but remember to prepare MSA (multiple sequence alignment) for each peptide before feeding them into Prospr.

The public classifiers can be found in:
* [CAMP3](http://www.camp3.bicnirrh.res.in/prediction.php): RF, DA, SVM
* [Scanner](https://www.dveltri.com/ascan/v2/ascan.html)
* [AMPMIC](https://github.com/zswitten/Antimicrobial-Peptides)
* [IAMPE](http://cbb1.ut.ac.ir/AMPClassifier/Index)
* [ampPEP](https://github.com/tlawrence3/amPEPpy)

Besides, feel free to use the *src/scripts/draw.py* for visulization. The detail description can be found in the script.
