@echo off  
chcp 65001 >nul  
  
del /q "%CD%\workspace\datasets\*"  
for /d %%p in ("%CD%\workspace\datasets\*") do ( rmdir /s/q %%p )  
