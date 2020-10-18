$Root = Resolve-Path -Path "${PSScriptRoot}\.."

$OutDir = "${Root}\output"

# 取得影片清單
$VideoList = @((Get-ChildItem -Path "${Root}\videos\*.mp4")) + @((Get-ChildItem -Path "${Root}\videos\*.MOV"))

Remove-Item -Path "$Env:TMP\*.mp4"
Remove-Item -Path "$Env:TMP\*.MOV"
foreach ($Video in $VideoList)
{
    $VName = $Video.Name
    Move-Item -Path "$($Video.FullName)" -Destination "$Env:TMP\$VName"
    if (-Not (Test-Path -Path "$Env:TMP\$VName")) { Exit }
    Start-Sleep -Milliseconds 100
    ffmpeg -hwaccel auto -i "$Env:TMP\$VName" -filter:v "crop=960:720:160:0" "$($Video.FullName)"
}


Remove-Item -Path "$Env:TMP\*.mp4"
Remove-Item -Path "$Env:TMP\*.MOV"
foreach ($Video in $VideoList)
{
    $VName = $Video.Name
    Move-Item -Path "$($Video.FullName)" -Destination "$Env:TMP\$VName"
    if (-Not (Test-Path -Path "$Env:TMP\$VName")) { Exit }
    Start-Sleep -Milliseconds 100
    ffmpeg -hwaccel auto -i "$Env:TMP\$VName" -vf scale=160:120 "$($Video.FullName)"
}
