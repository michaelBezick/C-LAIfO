# Variables
REMOTE = origin          # Change to your remote name (default is 'origin')
BRANCH = main            # Change to your branch name (default is 'main')
MESSAGE = "Auto commit"  # Default commit message

# Rules
all: push

add:
	git add --all

commit:
	git commit -m $(MESSAGE)

push: add commit
	git push $(REMOTE) $(BRANCH)

# Rule to specify a custom commit message
push-with-message:
	git add --all
	git commit -m "$(MESSAGE)"
	git push $(REMOTE) $(BRANCH)

# Rule to specify commit message as a parameter
push-message:
	@echo "Enter commit message: "; \
        read msg; \
	git add --all; \
	git commit -m "$$msg"; \
	git push $(REMOTE) $(BRANCH)

# Rule to run the training command
train:
	nohup python -u train_LAIL_MI.py task_agent=walker_walk task_expert=walker_walk agent=lail_cl_multiset difficulty=easy delta_source=0.2 delta_target=-0.25 apply_aug='CL-Q' aug_type='brightness' CL_data_type=agent batch_size=256 > output.log 2>&1 &

# Rule to check if the training script is running
check:
	ps -ef | grep train_LAIL_MI.py
