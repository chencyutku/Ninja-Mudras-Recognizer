$source_data_path = Resolve-Path -Path "${PSScriptRoot}\..\..\source_data"

if (-Not (Test-Path -Path "${source_data_path}\images"))
{ New-Item -Path "${source_data_path}" -Name "images" -ItemType Directory }

Get-ChildItem -Path "${PSScriptRoot}\output" | ForEach-Object -Process {
    Move-Item -Path $_ -Destination "${source_data_path}\images\$($_.BaseName)"
}

& "${source_data_path}\rename_data_pictures_with_label.ps1"
