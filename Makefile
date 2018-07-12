#!/usr/bin/env make -f
SHELL := /bin/bash
# Three things to add.
# 1. Multi gpu training
# 2. Persisting/reusing models
# 3. Tensorboard
# 4. All checkpoints copied to /export/a13/prastog3/
# 5. Create a method to compute the neural BS score.
# 6. Create a neural BS server, client scripts
FREE_GPU=$(shell free-gpu)
pwd:
	pwd
activate:
	echo source ~/tensorflow/bin/activate

# DATA_DIR := data/20news/
# DATA_DIR := data/rcv1-v2
DATA_DIR := data/tac2017
CHECKPOINT_DIR := checkpoint_$(notdir $(DATA_DIR))
VOCAB_SIZE := $(shell wc -l $(DATA_DIR)/vocab.new | cut -d' ' -f 1)
PORT := 12360
HOST := localhost
echo_%:
	echo $($*)

servei5:
    $(MAKE) DATA_DIR=../../data_dir/tac2017.i5 PORT=12342 serve

inspect:
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$(FREE_GPU) python inspect_saved_ckpt.py --vocab_size 63648

query:
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= ../../src/main/python/search-client.py $(HOST) $(PORT) --terms :Entity_ENG_EDL_0092354 :Entity_ENG_EDL_0023186 :Entity_ENG_EDL_0119473 --canonical_fn $(DATA_DIR)/entity.map --weights 2 -1 1

serve: $(DATA_DIR)/$(notdir $(DATA_DIR)).nvdm.pkl
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= python nvse.py --data_dir $(DATA_DIR)  --port $(PORT) --model_pkl $(notdir $(DATA_DIR)).nvdm.pkl --concrete_entity_dir $(DATA_DIR)/../tac2017.concrete.entities --k_rationale 30

test:
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$(FREE_GPU) python nvdm.py --test True --data_dir $(DATA_DIR) --checkpoint_dir $(CHECKPOINT_DIR) --vocab_size $(VOCAB_SIZE)

pickle:
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= python nvdm.py --data_dir $(DATA_DIR) --checkpoint_dir $(CHECKPOINT_DIR) --vocab_size $(VOCAB_SIZE) --model_pkl $(DATA_DIR)/$(notdir $(DATA_DIR)).nvdm.pkl

train:
	TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$(FREE_GPU) python nvdm.py --data_dir $(DATA_DIR) --checkpoint_dir $(CHECKPOINT_DIR) --vocab_size $(VOCAB_SIZE)

# | Epoch test: 1 | | Perplexity: 792.917282352 | Per doc ppx: 831.78333 | KLD: 25.466
# | Epoch test: 1 | | Perplexity: 590.525810452 | Per doc ppx: 497.49798 | KLD: 39.86
benchmark:
	TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES= python nvdm.py --test True --data_dir data/20news --checkpoint_dir ~/paper/nvbs/third_party/nvdm/checkpoint_20news/ --vocab_size 2000
	TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES= python nvdm.py --test True --data_dir data/20news --checkpoint_dir checkpoint_20news/ --vocab_size 2000

# TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= python nvdm.py --test True --data_dir data/rcv1-v2/ --checkpoint_dir ~/paper/nvbs/third_party/nvdm/checkpoint_rcv1-v2/ --vocab_size 10000
# TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= python nvdm.py --test True --data_dir data/rcv1-v2/ --checkpoint_dir checkpoint_rcv1-v2 --vocab_size 10000
