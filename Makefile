SHELL := /bin/bash

image:
	docker build docker-config -t auto-hamiltonian:latest

server:
	if command -v nvidia-smi &> /dev/null; then \
		echo "Running in GPU Mode"; \
		docker run --rm -it -p 8888:8888 \
		-v "$$PWD"/work:/work \
		-v "$$PWD"/docker-config/lab-config:/root/.jupyter/lab/user-settings \
		-v "$$PWD"/docker-config/zsh-config/.zshrc:/root/.zshrc \
		-v "$$PWD"/docker-config/nvim-config:/root/.config/nvim \
		--gpus all --pid=host \
		auto-hamiltonian:latest; \
	else \
		echo "Running in CPU Mode"; \
		docker run --rm -it -p 8888:8888 \
		-v "$$PWD"/work:/work \
		-v "$$PWD"/docker-config/lab-config:/root/.jupyter/lab/user-settings \
		-v "$$PWD"/docker-config/zsh-config/.zshrc:/root/.zshrc \
		-v "$$PWD"/docker-config/nvim-config:/root/.config/nvim \
		auto-hamiltonian:latest; \
	fi