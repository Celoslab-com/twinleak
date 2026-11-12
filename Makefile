# SPDX-License-Identifier: Apache-2.0
reproduce:
	python3 -m pip install -r requirements.txt
	python3 -m twinleak.reproduce
quick:
	python3 -m twinleak.reproduce --quick
