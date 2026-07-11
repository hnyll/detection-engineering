CLI := uv run python -m src.cli

.PHONY: setup data doctor new train eval tide review compare export parity bench gates interview phase0 publish

setup:            ## bootstrap uv + .venv from lockfile
	bash scripts/setup_env.sh

data:             ## symlink VisDrone into data/
	bash tools/setup_data.sh

doctor:           ## environment health checks
	$(CLI) doctor

new:              ## make new NAME=res1024  -> experiments/exp_NNN_res1024
	$(CLI) new $(NAME)

train:            ## make train EXP=exp_001_baseline [ARGS="--seeds 17,42,1337"]
	$(CLI) train $(EXP) $(ARGS)

eval:             ## make eval EXP=exp_001_baseline [ARGS="--seed 17"]
	$(CLI) eval $(EXP) $(ARGS)

tide:             ## make tide EXP=exp_001_baseline
	$(CLI) tide $(EXP) $(ARGS)

review:           ## make review EXP=exp_001_baseline ARGS="--launch"
	$(CLI) review $(EXP) $(ARGS)

compare:          ## make compare EXPS="exp_001_baseline exp_002_res1024"
	$(CLI) compare $(EXPS)

export:           ## make export EXP=... ARGS="--format onnx"
	$(CLI) export $(EXP) $(ARGS)

parity:           ## pt vs onnx numerical drift report
	$(CLI) parity $(EXP)

bench:            ## make bench EXP=... ARGS="--backend trt-fp16"
	$(CLI) bench $(EXP) $(ARGS)

gates:            ## competency exit-criteria report
	$(CLI) gates

interview:        ## generate interview companion skeletons
	$(CLI) interview $(ARGS)

phase0:           ## full COCO128 plumbing smoke test
	$(CLI) phase0

publish:          ## make publish ARGS="--init|--create-remote|--push"
	$(CLI) publish $(ARGS)
