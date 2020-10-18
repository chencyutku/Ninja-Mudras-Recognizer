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
  
powershell -ExecutionPolicy Bypass -File "scripts\split_video.ps1"  
