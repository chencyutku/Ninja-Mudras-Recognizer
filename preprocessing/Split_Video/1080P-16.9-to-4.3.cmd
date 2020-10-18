@echo off  
chcp 65001 >nul  
  
setlocal enabledelayedexpansion  
  
pushd "%~dp0..\.."  
set "ROOT=%CD%"  
(  
	endlocal  
	set "ROOT=%ROOT%"  
)  

set "PATH=%ROOT%\preprocessing\ffmpeg\bin;%PATH%"  
popd  

powershell -ExecutionPolicy Bypass -File "scripts\1080P-16.9-to-4.3.ps1"  

