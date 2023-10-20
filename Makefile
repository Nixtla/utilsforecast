.PHONY: all

info: header

define HEADER
  _    _ _   _ _     ______                           _   
 | |  | | | (_) |   |  ____|                         | |  
 | |  | | |_ _| |___| |__ ___  _ __ ___  ___ __ _ ___| |_ 
 | |  | | __| | / __|  __/ _ \| '__/ _ \/ __/ _` / __| __|
 | |__| | |_| | \__ \ | | (_) | | |  __/ (_| (_| \__ \ |_ 
  \____/ \__|_|_|___/_|  \___/|_|  \___|\___\__,_|___/\__|
UtilForecast is a utility package designed for Nixtla ecosystem.

Available commands:
* make setup: installs all requirements in `requirements.txt` and `nbdev_hooks`
* make test: runs `nbdev_clean && nbdev_export && mypy`

endef
export HEADER

header:
	clear
	@echo "$$HEADER"

setup:
	@python -m pip install -r ./dev/requirements.txt
	@nbdev_install_hooks

test:
	@nbdev_test
	@nbdev_clean
	@nbdev_export
	@mypy ./utilsforecast
