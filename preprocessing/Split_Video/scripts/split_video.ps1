$Root = Resolve-Path -Path "${PSScriptRoot}\.."

$OutDir = "${Root}\output"

# 取得影片清單
$VideoList = @((Get-ChildItem -Path "${Root}\videos\*.mp4")) + @((Get-ChildItem -Path "${Root}\videos\*.MOV"))

# 將每個影片以$FPS分拆為圖片，並輸出到各自Label的資料夾
$FPS = 3
foreach ($Video in $VideoList)
{
    ${OutFile_label} = $Video.BaseName
    if (-Not (Test-Path -Path "${OutDir}\${OutFile_label}"))
    {
        New-Item -Path "${OutDir}" -Name "${OutFile_label}" -ItemType Directory -Force
    }
    ffmpeg -hwaccel auto -i "$Video" -vf fps=$FPS "${OutDir}\${OutFile_label}\_%6d.jpg"
}

# 取得剛剛輸出的圖片所具備的Label清單
$LabelList = @((Get-ChildItem -Path "${OutDir}").FullName)

# 將頭尾去除，因為頭尾是操作攝影機的動作，僅保留中間的3/5
foreach ($LabelDir in $LabelList)
{
    $Picture_List = (Get-ChildItem -Path "$LabelDir").FullName
    $Picture_Num = $Picture_List.Count
    $Picture_List | ForEach-Object -Begin {
        $i = 0
    } -Process {
        if ($i -lt [Int64]($Picture_Num / 5)) { Remove-Item -Path $_ }
        if ($i -gt [Int64]($Picture_Num * 4 / 5)) { Remove-Item -Path $_ }
        $i++
    }
}


# 將提取完畢的圖片移動至程式提取用的來源資料夾
$source_data_path = Resolve-Path -Path "${Root}\..\..\source_data"

if (-Not (Test-Path -Path "${source_data_path}\images"))
{ New-Item -Path "${source_data_path}" -Name "images" -ItemType Directory }

(Get-ChildItem -Path "${Root}\output").FullName | ForEach-Object -Process {
    $Label = ([System.IO.FileInfo]$($_)).BaseName
    (Get-ChildItem -Path $_).FullName | ForEach-Object -Process {
        if (-Not (Test-Path -Path "${source_data_path}\images\${Label}"))
        {
            New-Item -Path "${source_data_path}\images" -Name "${Label}" -ItemType Directory -Force
        }
        Move-Item -Path "$_" -Destination "${source_data_path}\images\${Label}\$($_.BaseName)"
    }
}

& "${source_data_path}\rename_data_pictures_with_label.ps1"

Remove-Item -Path "${Root}\output" -Recurse -Force


# 將原始的影片依照日期備份
$timestamp = "$(Get-Date -format "yyyy.MMMM.dd__tt-hh-mm")"
New-Item -Path "$Root\videos_bak" -ItemType Directory -Name $timestamp
Move-Item -Path "$Root\videos\*" -Destination "$Root\videos_bak\$timestamp"
