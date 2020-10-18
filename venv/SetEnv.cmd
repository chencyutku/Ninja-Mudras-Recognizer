@echo off
chcp 65001 >nul

if not exist "%USERPROFILE%\.pyvenvs\NinjaMudras" (
	call :CREATE_ENV
)


call "%USERPROFILE%\.pyvenvs\NinjaMudras\Scripts\activate.bat"
python -m pip install -U pip
pip install -U setuptools
pip install -r package_requirements.txt

goto :eof














:CREATE_ENV

py -3.7 -m venv --prompt ninja "%USERPROFILE%\.pyvenvs\NinjaMudras"

exit /b 0 


