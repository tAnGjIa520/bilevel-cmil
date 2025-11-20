# Env
we use the python env similar to [CLAM](https://github.com/mahmoodlab/CLAM) and install the additional package from [MICIL](https://github.com/cvblab/MICIL) and [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)

# Data Prepare
For Skin Cancer Dataset, we downloaded the data from [MICIL](https://github.com/cvblab/MICIL). Please modify the code in `configs/csc_clam_cl.yaml` and `configs/csc_clam_cl.yaml` to set the data root.

For Camelyon and TCGA dataset, we followed [CLAM](https://github.com/mahmoodlab/CLAM). Please modify the code in `configs/c16_lung_rcc_clam_cl.yaml` and `configs/c16_lung_rcc_transmil_cl.yaml` to set the data root.

The data split files are stored in the `splits` dir. The folder already has a split group for Camelyon-TCGA; you can generate your own split files. The split code is from [CLAM](https://github.com/mahmoodlab/CLAM) and we modified it to fit the continual learning scenario.

# Training
For some config, you may need to modify the yaml file directly to make it take effect.
**Skin Cancer Dataset**
```shell
python main_cl.py --preset configs/csc_transmil_cl.yaml --cl_method prev --buffer_size 42 --exp_name csc_transmil_cl_buf42_attn_logit
python main_cl.py --preset configs/csc_clam_cl.yaml --cl_method prev --buffer_size 42 --exp_name csc_clam_cl_buf42_attn_logit
```
**Camelyon-TCGA**
```shell
python main_cl.py --preset configs/c16_lung_rcc_transmil_cl.yaml --cl_method prev --buffer_size 128 --buffer_slide_size 0.05 --distill_method maxrand --exp_name c16_lung_rcc_transmil_cl_pbbuf128_005_attn_logits_maxrand
python main_cl.py --preset configs/c16_lung_rcc_clam_cl.yaml --cl_method prev --buffer_size 128 --buffer_slide_size 0.05 --distill_method maxrand --exp_name c16_lung_rcc_clam_cl_pbbuf128_005_attn_logits_maxrand
```

You can find the results and logs in the `logs/<exp_name>` folder.